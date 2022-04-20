import argparse

import pytorch_lightning as pl

from pl_wandb_demo.data_modules import MNISTDataModule
from pl_wandb_demo.models import MNISTClassifier
from pl_wandb_demo.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)
    args = parser.parse_args()

    config = load_config(args.config)
    dm = MNISTDataModule(config=config["data_module_config"])
    model = MNISTClassifier(config=config["model_config"])

    trainer = pl.Trainer(**config["trainer_args"])
    trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()

