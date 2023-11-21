import torch.nn as nn
import torch


class DummyObjectRecognitionModel(nn.Module):
    def __init__(self, number_of_classes: int = 155):
        super().__init__()
        # here we can define model, but it's dummy model, so we don't need it here

    def forward(self, images: torch.Tensor):
        """
        :param images: (batch_size, 3, 640, 640)
        :return: random constant bboxes (xyxyn) and classes
        """
        batch_size = images.shape[0]
        batch_bboxes = []
        batch_labels = []
        for i in range(batch_size):
            bboxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])
            labels = torch.tensor([1, 2])
            batch_bboxes.append(bboxes)
            batch_labels.append(labels)
        return batch_bboxes, batch_labels
