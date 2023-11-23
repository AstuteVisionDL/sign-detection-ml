import pytorch_lightning as pl
import torch
from clearml import Task
from hydra.utils import instantiate

from sign_recognition.envs import settings
from hydra import main
from omegaconf import DictConfig


@main(config_path=f"{settings.PROJECT_DIR}/configs", config_name="default", version_base="1.2")
def train(config: DictConfig):
    pl.seed_everything(config.seed)
    task_name = f"{config.model.name}__{config.dataset.name}"
    Task.init(project_name=settings.PROJECT_NAME, task_name=task_name)
    # data
    data_module = instantiate(config.dataset.module)
    data_module.prepare_data()
    # model
    model = instantiate(config.model.module)
    # training
    trainer = pl.Trainer(max_epochs=config.max_epochs, limit_train_batches=0.001, limit_val_batches=0.005,
                         limit_test_batches=0.005)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    trainer.fit(model, train_loader, val_loader)
    # testing
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    dsize = config.dataset.module.dsize
    if config.save_onnx:
        model.to_onnx(f"{task_name}.onnx",
                      input_sample=torch.rand(1, 3, dsize, dsize),
                      export_params=True)
        Task.current_task().upload_artifact(f"{task_name}.onnx", artifact_object=f"{task_name}.onnx")


if __name__ == '__main__':
    train()
