import pytorch_lightning as pl
import torch

from sign_recognition.models.dummy.model import DummyObjectRecognitionModel


class DummyModelModule(pl.LightningModule):
    def __init__(self, number_of_classes: int = 155):
        super().__init__()
        self.model = DummyObjectRecognitionModel(number_of_classes)
        # here we can define model, but it's dummy model, so we don't need it here

    def forward(self, images: torch.Tensor):
        """
        :param images: (batch_size, 3, 640, 640)
        :return: random constant bboxes (xyxyn) and classes
        """
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        self(images)
        loss = torch.tensor(0.0)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        self(images)
        loss = torch.tensor(0.0)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
