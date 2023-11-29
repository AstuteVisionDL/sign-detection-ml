from pathlib import Path

import fiftyone as fo
import fiftyone.utils.coco as fouc
import torch
from PIL import Image

from .augmentations import Compose, PILToTensor


def collate_fn(batch):
    return tuple(zip(*batch, strict=True))


class RTSDDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        data_dir (Path): path to the dataset directory
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
    """

    def __init__(
        self,
        data_dir: Path,
        train: bool = True,
        transforms=None,
        gt_field="ground_truth",
        fast_dev_run: bool = False,
    ):
        split = "train" if train else "val"
        fifty_one_dataset = fo.Dataset.from_dir(dataset_dir=data_dir, dataset_type=fo.types.YOLOv5Dataset, split=split)
        if fast_dev_run:
            fifty_one_dataset = fifty_one_dataset.limit(100)
        fifty_one_dataset.compute_metadata()
        # FIXME: self.samples are mongo documents, so it's not pickleable and can't be used by Multiprocessing PyTorch's DataLoader (working only for num_workers = 0)
        self.samples = fifty_one_dataset
        self.transforms = transforms

        self.img_paths = self.samples.values("filepath")
        self.classes = self.samples.distinct("%s.detections.label" % gt_field)
        if self.classes[0] != "background":
            self.classes = ["background", *self.classes]
        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}
        self.gt_field = gt_field

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([idx]),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(image=img, target=target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes


if __name__ == "__main__":
    from sign_recognition.envs import settings
    processed_data_dir = settings.PROCESSED_DATA_PATH / "rtsd-dataset"

    test_transforms = Compose([PILToTensor()])
    rtsd_dataset = RTSDDataset(processed_data_dir, train=False, transforms=test_transforms, fast_dev_run=True)
    print(rtsd_dataset[0])
    data_loader = torch.utils.data.DataLoader(
        rtsd_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    for batch in data_loader:
        print(batch)
        break
