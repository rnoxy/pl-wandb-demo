import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.transform = Compose([ToTensor()])

    def setup(self, stage=None):
        from torchvision.datasets import MNIST

        self.train_set = MNIST(
            root="~/datasets", train=True, download=True, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
