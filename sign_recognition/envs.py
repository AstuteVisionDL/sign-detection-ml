from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # todo here will be config variables, which automatically loaded from .env file
    PROJECT_DIR: Path = Path(__file__).parent.parent
    RAW_DATA_PATH: Path = PROJECT_DIR / "data/raw"
    PROCESSED_DATA_PATH: Path = PROJECT_DIR / "data/processed"


settings = Settings()
