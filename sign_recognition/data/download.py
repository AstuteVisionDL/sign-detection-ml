from logging import getLogger
from pathlib import Path

from clearml import Dataset
from opendatasets import download

from sign_recognition.envs import settings

logger = getLogger(__name__)


def download_all_datasets(output_path: Path = settings.RAW_DATA_PATH):
    # download raw data from Kaggle and upload it to ClearML
    # here are preprocessing scripts for raw data (merging datasets, etc)
    # for training you should use already processed data from ClearML
    logger.info("Downloading datasets...")
    download("https://www.kaggle.com/datasets/watchman/rtsd-dataset", data_dir=output_path)
    logger.info("Dataset download complete.")
    dataset = Dataset.create(dataset_name="RTSD", dataset_project="SignTrafficRecognitionDL")
    dataset.add_files(output_path)
    dataset.upload()
    dataset.finalize()
    logger.info("Dataset upload complete.")


if __name__ == "__main__":
    download_all_datasets()
