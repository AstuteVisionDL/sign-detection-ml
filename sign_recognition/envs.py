from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # todo here will be config variables, which automatically loaded from .env file
    PROJECT_DIR: Path = Path(__file__).parent.parent

    MODELS_WEIGHTS_PATH: Path = PROJECT_DIR / "models"

    # ClearML Project name
    PROJECT_NAME: str = "SignTrafficRecognitionDL"
    # Default raw data path (original data from Kaggle)
    RAW_DATA_PATH: Path = PROJECT_DIR / "data/raw"
    # Default processed data path (data after preprocessing)
    PROCESSED_DATA_PATH: Path = PROJECT_DIR / "data/processed"
    PROCESSED_RTSD_DATASET_PATH: Path = PROJECT_DIR / "data/processed/rtsd-dataset"
    # Flag for uploading processed data to ClearML (if False, processed data will be saved only locally)
    UPLOAD_TO_CLEARML: bool = False


settings = Settings()
