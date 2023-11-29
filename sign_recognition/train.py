import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig

from envs import settings
from features.datamodule import RTSDDataModule
from models.faster_rcnn.module import FasterRCNNModule


@main(config_path=f"{settings.PROJECT_DIR}/configs", config_name="default", version_base="1.2")
def run_main(config: DictConfig) -> None:
    task_name = initialize_task(config)
    data_module = RTSDDataModule(num_workers=0,
                                 batch_size=config.dataset.module.batch_size,
                                 data_dir=settings.PROCESSED_RTSD_DATASET_PATH)
    data_module.prepare_data()
    data_module.setup("fit")

    config.model.module.number_of_classes = data_module.number_of_classes
    model = FasterRCNNModule(number_of_classes=config.model.module.number_of_classes)

    trainer = pl.Trainer(max_epochs=config.max_epochs)
    fit(data_module, model, trainer)
    test(data_module, trainer)
    save_model(config, model, task_name)


def initialize_task(config: DictConfig) -> str:
    pl.seed_everything(config.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    task_name = f"{config.model.name}__{config.dataset.name}"
    return task_name


def fit(data_module: pl.LightningDataModule, model: pl.LightningModule, trainer: pl.Trainer) -> None:
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    trainer.fit(model, train_loader, val_loader)


def test(data_module: pl.LightningDataModule, trainer: pl.Trainer) -> None:
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    trainer.test(dataloaders=test_loader, ckpt_path="best")


def save_model(config: DictConfig, model: pl.LightningModule, task_name: str) -> None:
    if config.save_onnx:
        dsize = config.dataset.module.dsize
        model.to_onnx(
            f"{settings.MODELS_WEIGHTS_PATH}/{task_name}.onnx",
            input_sample=torch.rand(1, 3, dsize, dsize),
            export_params=True,
        )


if __name__ == "__main__":
    run_main()
