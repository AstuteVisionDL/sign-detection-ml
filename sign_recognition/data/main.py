from logging import getLogger

from sign_recognition.data.download import download_dataset, upload_dataset_to_clearml
from sign_recognition.data.process import (
    export_to_yolov5,
    filter_classes,
    find_most_common_classes,
    to_fifty_one_dataset,
)
from sign_recognition.envs import settings

logger = getLogger(__name__)
logger.setLevel("INFO")


def create_processed_dataset(upload_to_clearml: bool = settings.UPLOAD_TO_CLEARML):
    raw_path = download_dataset()
    dataset = to_fifty_one_dataset(raw_path)
    classes = find_most_common_classes(dataset)
    dataset = filter_classes(dataset, classes)
    processed_path = export_to_yolov5(classes, dataset, output_path="/Users/arkadiysotnikov/PycharmProjects/sign-detection-ml/data/interim/rtsd-dataset")
    if upload_to_clearml:
        upload_dataset_to_clearml(processed_path)


if __name__ == "__main__":
    create_processed_dataset()
