import pytorch_lightning as pl
from clearml import Task

from sign_recognition.features.datamodule import RTSDDataModule
from sign_recognition.models.faster_rcnn.module import FasterRCNNModule

if __name__ == "__main__":
    pl.seed_everything(42)

    # todo need to be specified in config
    batch_size = 4
    max_epochs = 3

    Task.init(project_name="SignTrafficRecognitionDL", task_name="fasterrcnn_mobilenet_v3_large_fpn")

    # ------------
    # data
    # ------------
    data_module = RTSDDataModule(batch_size)
    data_module.prepare_data()

    # ------------
    # model
    # ------------
    model = FasterRCNNModule()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(max_epochs=max_epochs, limit_train_batches=0.001, limit_val_batches=0.005, limit_test_batches=0.005)
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
