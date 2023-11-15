
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


def download_all_datasets(output_path: Path):
    logger.info("Downloading datasets...")
    # todo download datasets here
    logger.info("Dataset download complete.")
