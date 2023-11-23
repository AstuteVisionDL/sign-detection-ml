from typing import Literal

import clearml
import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics import Accuracy, Precision, Recall, F1Score


def build_metrics_dict(
    stage: Literal["train", "val", "test"],
    num_classes: int
):
    task: Literal["binary", "multiclass", "multilabel"] = "multilabel"
    metrics_dict = {f"{stage}_acc": Accuracy(task=task, num_labels=num_classes),
                    f"{stage}_precision": Precision(task=task, num_labels=num_classes),
                    f"{stage}_recall": Recall(task=task, num_labels=num_classes),
                    f"{stage}_f1": F1Score(task=task, num_labels=num_classes),
                    f"{stage}_precision_weighted": Precision(task=task, num_labels=num_classes, average="weighted"),
                    f"{stage}_recall_weighted": Recall(task=task, num_labels=num_classes, average="weighted"),
                    f"{stage}_f1_weighted": F1Score(task=task, num_labels=num_classes, average="weighted")}
    return metrics_dict


def convert_to_torchmetrics_format(labels, predictions, number_of_classes):
    pred_labels = []
    for prediction in predictions:
        pred_labels.append(prediction['labels'])
    # convert to format required by torchmetrics (one hot encoded)
    # we don't take into account the possible repeated labels in the same image because main goal is to notify the
    # driver of the presence of a sign, not to count or detect them
    for i in range(len(pred_labels)):
        pred_labels[i] = torch.nn.functional.one_hot(pred_labels[i], num_classes=number_of_classes).sum(dim=0)
        labels[i] = torch.nn.functional.one_hot(labels[i], num_classes=number_of_classes).sum(dim=0)
        pred_labels[i][pred_labels[i] > 1] = 1
        labels[i][labels[i] > 1] = 1
    pred_labels = torch.stack(pred_labels)
    labels = torch.stack(labels)
    return labels, pred_labels


def build_model(number_of_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        pretrained=True
    )
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_of_classes)
    return model


class FasterRCNNModule(pl.LightningModule):
    def __init__(self, number_of_classes: int = 155, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters("number_of_classes", "learning_rate")
        self.model = build_model(number_of_classes)

        self.val_metrics_dict = build_metrics_dict("val", number_of_classes)
        self.test_metrics_dict = build_metrics_dict("test", number_of_classes)

    def forward(self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]]):
        """
        :param targets: list of dicts with keys 'boxes' and 'labels'
        :param images: (batch_size, 3, 640, 640)
        :return: dict with keys 'boxes' and 'labels' or loss dict if targets are provided
        """
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        # batch = (images, bboxes, labels)
        # images = (batch_size, 3, dsize(0), dsize(1)) = (batch_size, 3, 640, 640) by default
        # bboxes = (batch_size, number_of_bboxes, 4) in xyxyn format
        # labels = (batch_size, number_of_bboxes)
        images, bboxes, labels = batch
        images = list(image for image in images)
        targets = [{"boxes": bboxes[i], "labels": labels[i]} for i in range(len(images))]

        loss_dict = self(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses)
        clearml.Logger.current_logger().report_scalar("train_loss", "train_loss", losses, self.global_step)
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        images = list(image for image in images)
        targets = [{"boxes": bboxes[i], "labels": labels[i]} for i in range(len(images))]
        predictions = self(images, targets)
        labels, pred_labels = convert_to_torchmetrics_format(labels, predictions, self.hparams.number_of_classes)
        for metric in self.val_metrics_dict.values():
            metric.update(pred_labels, labels)

    def on_validation_epoch_end(self):
        for metric_name, metric in self.val_metrics_dict.items():
            self.log(metric_name, metric.compute())
            clearml.Logger.current_logger().report_scalar(metric_name, metric_name, metric.compute(), self.current_epoch)
            metric.reset()

    def test_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        targets = [{"boxes": bboxes[i], "labels": labels[i]} for i in range(len(images))]
        predictions = self(images, targets)
        labels, pred_labels = convert_to_torchmetrics_format(labels, predictions, self.hparams.number_of_classes)
        for metric in self.test_metrics_dict.values():
            metric.update(pred_labels, labels)

    def on_test_epoch_end(self):
        for metric_name, metric in self.test_metrics_dict.items():
            self.log(metric_name, metric.compute())

    def configure_optimizers(self):
        print(list(self.named_parameters()))
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
