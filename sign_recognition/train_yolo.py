import hydra
from clearml import Task
from hydra.utils import instantiate

from sign_recognition.envs import settings
from ultralytics import YOLO
from omegaconf import DictConfig


@hydra.main(config_path=f"{settings.PROJECT_DIR}/configs", config_name="yolo_default", version_base="1.2")
def train_yolo(config: DictConfig):
    """
    Train YOLO model for sign recognition. We don't include this code in the main training script,
    because we want to train YOLO separately from other models.
    YOLO framework is not supported by PyTorch Lightning, so we can't use it in the main training script:
    We don't have access for training step in YOLO, so we can't use Pytorch Lightning for training.
    Returns:

    """
    model: YOLO = instantiate(config.model.module)
    results = model.train(
        data=settings.PROCESSED_RTSD_DATASET_PATH / "dataset.yaml",
        epochs=config["max_epochs"],
        batch=config.dataset.module.batch_size,
        workers=config.dataset.module.num_workers,
        patience=config.early_stop_patience,
        imgsz=config.dataset.module.dsize,
        project=settings.PROJECT_NAME,
        plots=True,
        name=f"{config.model.name}__{config.dataset.name}"
    )
    print(results)
    Task.current_task().connect(config)
    model.save(settings.MODELS_WEIGHTS_PATH / "yolo.pt")
    print(model.val())
    print(model.benchmark())
    model.export(format="onnx", f=settings.MODELS_WEIGHTS_PATH / "yolo.onnx")


if __name__ == '__main__':
    train_yolo()
