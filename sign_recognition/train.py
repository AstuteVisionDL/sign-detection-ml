import pytorch_lightning as pl
import torch
from clearml import Task

from sign_recognition.envs import settings
from sign_recognition.features.datamodule import RTSDDataModule
from sign_recognition.models.faster_rcnn.module import FasterRCNNModule


def train():
    pl.seed_everything(42)

    # todo need to be specified in config
    task_name = "fasterrcnn_mobilenet_v3_large_fpn"
    batch_size = 8
    max_epochs = 1
    dsize = (640, 640)

    Task.init(project_name="SignTrafficRecognitionDL", task_name=task_name)

    # ------------
    # data
    # ------------
    data_module = RTSDDataModule(batch_size, dsize=dsize)
    data_module.prepare_data()

    # ------------
    # model
    # ------------
    model = FasterRCNNModule()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(max_epochs=max_epochs, limit_train_batches=0.001, limit_val_batches=0.005,
                         limit_test_batches=0.005)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    trainer.test(dataloaders=test_loader, ckpt_path='best')

    # ------------
    # inference
    # ------------
    # convert to onnx
    model.to_onnx(f"{task_name}.onnx",
                  input_sample=torch.rand(batch_size, 3, dsize[0], dsize[1]),
                  export_params=True)


if __name__ == '__main__':
    train()