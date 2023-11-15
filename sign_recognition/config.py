from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pathlib import Path

load_dotenv()


class Settings(BaseSettings):
    # todo here will be config variables, which automatically loaded from .env file
    RAW_DATA_PATH: Path = "./data/raw"
    PROCESSED_DATA_PATH: Path = "./data/processed"


settings = Settings()
