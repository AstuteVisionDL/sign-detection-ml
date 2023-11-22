import clearml
import pytorch_lightning as pl
import torch
from sign_recognition.models.dummy.model import DummyObjectRecognitionModel
from torchmetrics import Accuracy, Precision, Recall, F1Score


class DummyModelModule(pl.LightningModule):
    def __init__(self, number_of_classes: int = 155, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.model = DummyObjectRecognitionModel(number_of_classes)
        self.save_hyperparameters("hidden_dim", "learning_rate")
        self.val_acc = Accuracy(task="multiclass", num_classes=number_of_classes)
        self.val_precision = Precision(task="multiclass", num_classes=number_of_classes)
        self.val_recall = Recall(task="multiclass", num_classes=number_of_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=number_of_classes)

        self.test_acc = Accuracy(task="multiclass", num_classes=number_of_classes)
        self.test_precision = Precision(task="multiclass", num_classes=number_of_classes)
        self.test_recall = Recall(task="multiclass", num_classes=number_of_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=number_of_classes)

    def forward(self, images: torch.Tensor):
        """
        :param images: (batch_size, 3, 640, 640)
        :return: random constant bboxes (xyxyn) and classes
        """
        return self.model(images)

    def training_step(self, batch, batch_idx):
        # batch = (images, bboxes, labels)
        # images = (batch_size, 3, dsize(0), dsize(1)) = (batch_size, 3, 640, 640) by default
        # bboxes = (batch_size, number_of_bboxes, 4) in xyxyn format
        # labels = (batch_size, number_of_bboxes)
        images, bboxes, labels = batch
        self(images)
        loss = torch.tensor(0.0, requires_grad=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        pred_bboxes, pred_labels = self(images)
        pred_labels = torch.cat(pred_labels)
        labels = torch.cat(labels)
        if len(labels) != len(pred_labels):
            # pad with -1
            pred_labels = torch.cat([pred_labels, torch.tensor([-1] * (len(labels) - len(pred_labels)))])
            labels = torch.cat([labels, torch.tensor([-1] * (len(pred_labels) - len(labels)))])
        self.val_acc.update(pred_labels, labels)
        self.val_precision.update(pred_labels, labels)
        self.val_recall.update(pred_labels, labels)
        self.val_f1.update(pred_labels, labels)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        self.log("val_acc", val_acc)
        clearml.Logger.current_logger().report_scalar("val_acc", "val_acc", val_acc, self.current_epoch)
        self.val_acc.reset()

        val_precision = self.val_precision.compute()
        self.log("val_precision", val_precision)
        clearml.Logger.current_logger().report_scalar("val_precision", "val_precision", val_precision, self.current_epoch)
        self.val_precision.reset()

        val_recall = self.val_recall.compute()
        self.log("val_recall", val_recall)
        clearml.Logger.current_logger().report_scalar("val_recall", "val_recall", val_recall, self.current_epoch)
        self.val_recall.reset()

        val_f1 = self.val_f1.compute()
        self.log("val_f1", val_f1)
        clearml.Logger.current_logger().report_scalar("val_f1", "val_f1", val_f1, self.current_epoch)
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        pred_bboxes, pred_labels = self(images)
        pred_labels = torch.cat(pred_labels)
        labels = torch.cat(labels)
        if len(labels) != len(pred_labels):
            # pad with -1
            pred_labels = torch.cat([pred_labels, torch.tensor([-1] * (len(labels) - len(pred_labels)))])
            labels = torch.cat([labels, torch.tensor([-1] * (len(pred_labels) - len(labels)))])
        self.test_acc.update(pred_labels, labels)
        self.test_precision.update(pred_labels, labels)
        self.test_recall.update(pred_labels, labels)
        self.test_f1.update(pred_labels, labels)

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        self.log("test_acc", test_acc)
        clearml.Logger.current_logger().report_scalar("test_acc", "test_acc", test_acc, self.current_epoch)
        self.test_acc.reset()

        test_precision = self.test_precision.compute()
        self.log("test_precision", test_precision)
        clearml.Logger.current_logger().report_scalar("test_precision", "test_precision", test_precision, self.current_epoch)
        self.test_precision.reset()

        test_recall = self.test_recall.compute()
        self.log("test_recall", test_recall)
        clearml.Logger.current_logger().report_scalar("test_recall", "test_recall", test_recall, self.current_epoch)
        self.test_recall.reset()

        test_f1 = self.test_f1.compute()
        self.log("test_f1", test_f1)
        clearml.Logger.current_logger().report_scalar("test_f1", "test_f1", test_f1, self.current_epoch)
        self.test_f1.reset()

    def configure_optimizers(self):
        # Check if there are any parameters in your model
        print(list(self.parameters()))
        print(list(self.named_parameters()))
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
