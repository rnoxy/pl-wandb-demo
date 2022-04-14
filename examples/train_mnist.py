import pytorch_lightning as pl
from pl_wandb_demo.data_modules import MNISTDataModule
from pl_wandb_demo.models import MNISTClassifier


def main():
    dm = MNISTDataModule(batch_size=32)
    model = MNISTClassifier(units=100)

    trainer = pl.Trainer(gpus=-1, max_epochs=10)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
