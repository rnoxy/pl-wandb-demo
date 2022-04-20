import pytorch_lightning as pl
from pl_wandb_demo.data_modules import MNISTDataModule
from pl_wandb_demo.models import MNISTClassifier
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    dm = MNISTDataModule(batch_size=32)
    model = MNISTClassifier.load_from_checkpoint(checkpoint_path=args.model_path)

    trainer = pl.Trainer(gpus=-1, max_epochs=25)
    trainer.fit(model, datamodule=dm, ckpt_path=args.model_path)


if __name__ == "__main__":
    main()
