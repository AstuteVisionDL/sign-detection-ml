import clearml
import pytorch_lightning as pl
import torch

from sign_recognition.models.faster_rcnn.model import build_model


class FasterRCNNModule(pl.LightningModule):
    def __init__(self, number_of_classes: int, learning_rate: float = 1e-3, activate_clearml: bool = True) -> None:
        super().__init__()
        self.save_hyperparameters("number_of_classes", "learning_rate")
        print(self.hparams)
        self.model = build_model(number_of_classes)
        self.clearml = activate_clearml

    def forward(self, images: torch.Tensor) -> dict:
        """
        :param images: (batch_size, 3, 640, 640)
        :return: dict with keys 'boxes' and 'labels' or loss dict if targets are provided
        """
        return self.model(images)

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        images, targets = batch
        self.model.train()
        loss_dict = self.model(images, targets)
        losses = 0
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train_{loss_name}", loss_value)
            losses += loss_value
            if self.clearml:
                clearml.Task.current_task().get_logger().report_scalar(
                    f"train_{loss_name}", "train", self.current_epoch, int(loss_value)
                )
        return {"loss": losses}

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, targets = batch
        self.model.eval()
        predictions = self.model(images)
        print(predictions)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        images, targets = batch
        self.model.eval()
        predictions = self.model(images)
        print(predictions)

    def on_fit_end(self) -> None:
        self.model.eval()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    from sign_recognition.envs import settings
    from sign_recognition.features.datamodule import RTSDDataModule

    data_dir = settings.PROCESSED_DATA_PATH / "rtsd-dataset"
    data_module = RTSDDataModule(data_dir=data_dir, activate_clearml=False)
    data_module.prepare_data()
    data_module.setup("test")
    model = FasterRCNNModule(data_module.number_of_classes)
    model.training_step(data_module.test_dataloader().__iter__().__next__(), 0)
    model.validation_step(data_module.test_dataloader().__iter__().__next__(), 0)
    model.test_step(data_module.test_dataloader().__iter__().__next__(), 0)
