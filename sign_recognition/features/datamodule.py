import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import tqdm
from clearml import Dataset
from torch.utils.data import DataLoader, Subset, random_split

from sign_recognition.envs import settings
from sign_recognition.features.augmentations import Compose, Normalize, PILToTensor, RandomHorizontalFlip, Resize
from sign_recognition.features.dataset import RTSDDataset, collate_fn

logger = logging.getLogger(__name__)


class RTSDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        dsize: int = 640,
        data_dir: Path | None = settings.PROCESSED_DATA_PATH / "rtsd-dataset",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.rtsd_train: Subset | None = None
        self.rtsd_val: Subset | None = None
        self.rtsd_test: Subset | None = None
        self.batch_size = batch_size
        self.train_transforms = Compose(
            [
                PILToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                Resize(dsize),
                RandomHorizontalFlip(0.5),
            ]
        )
        self.seed = seed

    def prepare_data(self) -> None:
        if self.data_dir is None:
            # download data from ClearML
            data_dir = Dataset.get(
                dataset_project="SignTrafficRecognitionDL", dataset_name="RTSD", alias="RTSD"
            ).get_local_copy()
            self.data_dir = Path(data_dir) / "rtsd-dataset"
        else:
            logger.info(f"Data dir is {self.data_dir}")

    def setup(self, stage: str) -> None:
        logger.info(f"Setup data module with stage {stage}")
        if stage == "fit" or stage is None:
            rtsd_full = RTSDDataset(self.data_dir, train=True, transforms=self.train_transforms)
            train_size = int(len(rtsd_full) * 0.8)
            val_size = len(rtsd_full) - train_size
            self.rtsd_train, self.rtsd_val = random_split(
                rtsd_full, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
            )
        if stage == "test":
            test_transforms = Compose(
                [
                    PILToTensor(),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    Resize(640),
                ]
            )
            self.rtsd_test = RTSDDataset(self.data_dir, train=False, transforms=test_transforms)
        logger.info(f"Setup data module with stage {stage} finished")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.rtsd_train, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.rtsd_val, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.rtsd_test, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn)

    @property
    def number_of_classes(self) -> int:
        if self.rtsd_train is None and self.rtsd_test is None:
            raise ValueError("Data module is not initialized")
        if self.rtsd_train is None:
            logger.warning("Data module is initialized only for test stage")
            return len(self.rtsd_test.get_classes())
        return len(self.rtsd_train.dataset.get_classes())


if __name__ == "__main__":
    data_module = RTSDDataModule(data_dir=settings.PROCESSED_DATA_PATH / "rtsd-dataset")
    data_module.prepare_data()
    data_module.setup("test")
    for batch in tqdm.tqdm(data_module.test_dataloader()):
        print(batch)
        break
