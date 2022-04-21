import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from pl_wandb_demo.data_modules import MNISTDataModule
from pl_wandb_demo.models import MNISTClassifier
from pl_wandb_demo.utils import load_config


def get_callbacks():

    model_checkpoint_callback = ModelCheckpoint(
        dirpath="data/checkpoints",
        filename="mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        monitor="val/loss", mode="min",
        auto_insert_metric_name=False,
        verbose=True
    )

    return [
        model_checkpoint_callback,
        RichModelSummary(),
        RichProgressBar()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)
    args = parser.parse_args()

    config = load_config(args.config)
    dm = MNISTDataModule(config=config["data_module_config"])
    model = MNISTClassifier(config=config["model_config"])

    callbacks = get_callbacks()
    logger = WandbLogger(offline=True)

    trainer = pl.Trainer(**config["trainer_args"], callbacks=callbacks, logger=WandbLogger())
    trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
