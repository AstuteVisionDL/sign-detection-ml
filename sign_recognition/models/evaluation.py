from typing import Literal

import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall


def build_metrics_dict(stage: Literal["train", "val", "test"], num_classes: int, device) -> dict:
    task: Literal["binary", "multiclass", "multilabel"] = "multilabel"
    metrics_dict = {
        f"{stage}_acc": Accuracy(task=task, num_labels=num_classes).to(device),
        f"{stage}_precision": Precision(task=task, num_labels=num_classes).to(device),
        f"{stage}_recall": Recall(task=task, num_labels=num_classes).to(device),
        f"{stage}_f1": F1Score(task=task, num_labels=num_classes).to(device),
        f"{stage}_precision_weighted": Precision(task=task, num_labels=num_classes, average="weighted").to(device),
        f"{stage}_recall_weighted": Recall(task=task, num_labels=num_classes, average="weighted").to(device),
        f"{stage}_f1_weighted": F1Score(task=task, num_labels=num_classes, average="weighted").to(device),
    }
    return metrics_dict


def convert_to_torchmetrics_format(
    labels: list, predictions: dict, number_of_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_labels = []
    for prediction in predictions:
        pred_labels.append(prediction["labels"])
    # convert to format required by torchmetrics (one hot encoded)
    # we don't take into account the possible repeated labels in the same image because main goal is to notify the
    # driver of the presence of a sign, not to count or detect them
    for i in range(len(pred_labels)):
        pred_labels[i] = torch.nn.functional.one_hot(pred_labels[i], num_classes=number_of_classes).sum(dim=0)
        labels[i] = torch.nn.functional.one_hot(labels[i], num_classes=number_of_classes).sum(dim=0)
        pred_labels[i][pred_labels[i] > 1] = 1
        labels[i][labels[i] > 1] = 1
    return torch.stack(labels), torch.stack(pred_labels)
