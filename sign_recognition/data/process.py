import os
from pathlib import Path
from typing import Literal
from logging import getLogger

import fiftyone as fo
from fiftyone import ViewField as F

from sign_recognition.envs import settings

import shutil


logger = getLogger(__name__)


def to_fifty_one_dataset(raw_dataset_path: Path, dataset_name: str = "RTSD"):
    rename_frames_dir(raw_dataset_path)
    rename_annotation_json(raw_dataset_path, subset="train")
    fifty_one_dataset = fo.Dataset(dataset_name)
    fifty_one_dataset.add_dir(
        dataset_dir=raw_dataset_path,
        dataset_type=fo.types.COCODetectionDataset,
        tags="train",
    )
    rename_annotation_json(raw_dataset_path, subset="val")
    fifty_one_dataset.add_dir(
        dataset_dir=raw_dataset_path,
        dataset_type=fo.types.COCODetectionDataset,
        tags="val",
    )
    print("Dataset converting complete.")
    print(f"Dataset summary: {fifty_one_dataset}")
    print(f"Dataset samples head: {fifty_one_dataset.head(10)}")
    return fifty_one_dataset


def rename_frames_dir(raw_dataset_path: Path):
    if os.path.exists(raw_dataset_path / "data"):
        logger.warning("Frames directory already renamed.")
        return
    os.makedirs(raw_dataset_path / "data", exist_ok=True)
    os.rename(raw_dataset_path / "rtsd-frames", raw_dataset_path / "data")


def rename_annotation_json(raw_dataset_path: Path,
                           subset: Literal['train', 'val'] = "train"):
    if os.path.exists(raw_dataset_path / "labels.json"):
        os.remove(raw_dataset_path / "labels.json")
    shutil.copy(raw_dataset_path / f"{subset}_anno.json", raw_dataset_path / "labels.json")


def find_most_common_classes(dataset):
    dataset.compute_metadata()
    # find most common classes
    labels_statistics: dict[str, int] = dataset.count_values("detections.detections.label")
    # find most common classes
    include_classes = set()
    one_percent = dataset.count("detections.detections.label") * 0.01
    for label, count in labels_statistics.items():
        if count > one_percent:
            include_classes.add(label)
    return include_classes


def filter_classes(dataset, include_classes):
    filtered_view = dataset.filter_labels(
        "detections", F("label").is_in(include_classes)
    )
    return filtered_view


def export_to_yolov5(classes, dataset, output_path=settings.PROCESSED_DATA_PATH / "rtsd-dataset", overwrite=False):
    # filter by split (train/val)
    output_path = str(output_path)
    if os.path.exists(output_path) and not overwrite:
        logger.warning("Dataset already exported. Use overwrite=True to overwrite.")
        return output_path
    train_view = dataset.match(F("tags").contains("train"))
    val_view = dataset.match(F("tags").contains("val"))
    # export view to yolov5 format
    train_view.export(
        export_dir=output_path,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="detections",
        classes=classes,
        overwrite=True,
        split="train",
    )
    val_view.export(
        export_dir=output_path,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="detections",
        classes=classes,
        split="val",
    )
    return output_path
