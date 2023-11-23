import json
from collections import defaultdict
from pathlib import Path

import albumentations
import clearml
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data.dataset import Dataset


def collate_rtsd_fn(batch):
    """
    Since each image may have a different number of objects,
    we need a collate function (to be passed to the DataLoader).

    This describes how to combine these tensors of different sizes. We use lists.

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
    """

    images = []
    boxes = []
    labels = []

    for sample in batch:
        images.append(sample[0])
        boxes.append(sample[1])
        labels.append(sample[2])

    images = torch.stack(images, dim=0)

    return images, boxes, labels


class DatasetRTSD(Dataset):
    def __init__(self, data_path: Path, train=True, transform=None) -> None:
        super().__init__()
        with open(data_path / "label_map.json") as f:
            self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
        annotation_path = data_path / "train_anno.json" if train else data_path / "val_anno.json"
        with open(annotation_path) as f:
            self.annotations = json.load(f)
        self.frames_dir = data_path / "rtsd-frames"
        self.image_id2annotations = defaultdict(list)
        for annotation in self.annotations["annotations"]:
            image_id = annotation["image_id"]
            self.image_id2annotations[image_id].append(annotation)
        if transform is None:
            # assign default augmentations
            self.transform = albumentations.Compose(
                [
                    albumentations.Resize(width=640, height=640),
                    albumentations.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = transform

    def __getitem__(self, idx):
        """

        :param idx: index of the image
        :return: tuple of image, list of bounding boxes in xyxyn format,
        list of labels indexes
        """
        image_path = self.frames_dir / self.annotations["images"][idx]["file_name"]
        image_id = self.annotations["images"][idx]["id"]
        width = self.annotations["images"][idx]["width"]
        height = self.annotations["images"][idx]["height"]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        annotations = self.image_id2annotations[image_id]
        bboxes = []
        labels = []

        for annotation in annotations:
            bbox_xywh = annotation["bbox"]
            bbox_xyxyn = [
                bbox_xywh[0] / width,
                bbox_xywh[1] / height,
                (bbox_xywh[0] + bbox_xywh[2]) / width,
                (bbox_xywh[1] + bbox_xywh[3]) / height,
            ]
            label = annotation["category_id"]
            bboxes.append(bbox_xyxyn)
            labels.append(label)

        return image, torch.tensor(bboxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.annotations["images"])


if __name__ == "__main__":
    data_dir = clearml.Dataset.get(dataset_project="SignTrafficRecognitionDL", dataset_name="RTSD").get_local_copy()
    data_dir = Path(data_dir) / "rtsd-dataset"
    print(data_dir)
    dataset = DatasetRTSD(data_dir)
    counter = 0
    for data in dataset:
        print(data[0])
        print(len(data[1]))
        if counter > 10:
            break
