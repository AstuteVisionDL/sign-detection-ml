import clearml
import pytorch_lightning as pl
import torch

from sign_recognition.models.evaluation import build_metrics_dict, convert_to_torchmetrics_format
from sign_recognition.models.faster_rcnn.model import build_model, convert_to_fastercnn_format


class FasterRCNNModule(pl.LightningModule):
    def __init__(self, number_of_classes: int = 155, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters("number_of_classes", "learning_rate")
        self.model = build_model(number_of_classes)

        self.val_metrics_dict = build_metrics_dict("val", number_of_classes)
        self.test_metrics_dict = build_metrics_dict("test", number_of_classes)

    def forward(self, images: torch.Tensor):
        """
        :param images: (batch_size, 3, 640, 640)
        :return: dict with keys 'boxes' and 'labels' or loss dict if targets are provided
        """
        return self.model(images)

    def training_step(self, batch, batch_idx):
        # batch = (images, bboxes, labels)
        # images = (batch_size, 3, dsize(0), dsize(1)) = (batch_size, 3, 640, 640) by default
        # bboxes = (batch_size, number_of_bboxes, 4) in xyxyn format
        # labels = (batch_size, number_of_bboxes)
        images, bboxes, labels = batch
        targets = convert_to_fastercnn_format(bboxes, images, labels)
        self.model.train()
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses)
        clearml.Logger.current_logger().report_scalar("train_loss", "train_loss", losses, self.global_step)
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        images = list(image for image in images)
        self.model.eval()
        predictions = self.model(images)
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
        self.model.eval()
        predictions = self.model(images)
        labels, pred_labels = convert_to_torchmetrics_format(labels, predictions, self.hparams.number_of_classes)
        for metric in self.test_metrics_dict.values():
            metric.update(pred_labels, labels)

    def on_test_epoch_end(self):
        for metric_name, metric in self.test_metrics_dict.items():
            self.log(metric_name, metric.compute())

    def on_fit_end(self) -> None:
        self.model.eval()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
