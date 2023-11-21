import logging
from pathlib import Path

import albumentations
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
import tqdm
from sign_recognition.features.dataset import DatasetRTSD, collate_rtsd_fn
from clearml import Dataset


logger = logging.getLogger(__name__)


class RTSDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, dsize=(640, 640)):
        super().__init__()
        self.data_dir = None
        self.rtsd_train = None
        self.rtsd_val = None
        self.rtsd_test = None
        self.batch_size = batch_size
        self.transform = albumentations.Compose([
            albumentations.Resize(width=dsize[0], height=dsize[1]),
            # todo calculate mean and std for our dataset
            albumentations.Normalize(),
            ToTensorV2(),
        ])

    def prepare_data(self) -> None:
        # download data from ClearML
        data_dir = Dataset.get(dataset_project='SignTrafficRecognitionDL', dataset_name='RTSD').get_local_copy()
        self.data_dir = Path(data_dir) / "rtsd-dataset"

    def setup(self, stage: str):
        logger.info(f"Setup data module with stage {stage}")
        if stage == "fit" or stage is None:
            rtsd_full = DatasetRTSD(self.data_dir, train=True, transform=self.transform)
            train_size = int(len(rtsd_full) * 0.8)
            val_size = len(rtsd_full) - train_size
            self.rtsd_train, self.rtsd_val = random_split(rtsd_full,
                                                          [train_size, val_size],
                                                          generator=torch.Generator().manual_seed(42))
        if stage == "test":
            self.rtsd_test = DatasetRTSD(self.data_dir, train=False, transform=self.transform)
        logger.info(f"Setup data module with stage {stage} finished")

    def train_dataloader(self):
        return DataLoader(self.rtsd_train, batch_size=self.batch_size, collate_fn=collate_rtsd_fn)

    def val_dataloader(self):
        return DataLoader(self.rtsd_val, batch_size=self.batch_size, collate_fn=collate_rtsd_fn)

    def test_dataloader(self):
        return DataLoader(self.rtsd_test, batch_size=self.batch_size, collate_fn=collate_rtsd_fn)


if __name__ == '__main__':
    data_module = RTSDDataModule()
    data_module.prepare_data()
    data_module.setup("fit")
    for batch in tqdm.tqdm(data_module.train_dataloader()):
        print(batch)
