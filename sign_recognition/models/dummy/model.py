import torch
import torch.nn as nn


class DummyObjectRecognitionModel(nn.Module):
    def __init__(self, number_of_classes: int = 155):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 640 * 640, number_of_classes)

    def forward(self, images: torch.Tensor):
        """
        :param images: (batch_size, 3, 640, 640)
        :return: random constant bboxes (xyxyn) and classes
        """
        x = self.conv(images)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        batch_size = images.shape[0]
        batch_bboxes = []
        batch_labels = []
        for _i in range(batch_size):
            bboxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])
            labels = torch.tensor([1, 2])
            batch_bboxes.append(bboxes)
            batch_labels.append(labels)
        return batch_bboxes, batch_labels
