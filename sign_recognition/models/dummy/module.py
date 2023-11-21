import pytorch_lightning as pl
import torch

from sign_recognition.models.dummy.model import DummyObjectRecognitionModel


class DummyModelModule(pl.LightningModule):
    def __init__(self, number_of_classes: int = 155, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.model = DummyObjectRecognitionModel(number_of_classes)
        self.save_hyperparameters("hidden_dim", "learning_rate")

    def forward(self, images: torch.Tensor):
        """
        :param images: (batch_size, 3, 640, 640)
        :return: random constant bboxes (xyxyn) and classes
        """
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        self(images)
        loss = torch.tensor(0.0, requires_grad=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        self(images)
        loss = torch.tensor(0.0)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        self(images)
        loss = torch.tensor(0.0)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        # Check if there are any parameters in your model
        print(list(self.parameters()))
        print(list(self.named_parameters()))
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
