import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.transform = Compose([ToTensor()])

    def setup(self, stage=None):
        from torchvision.datasets import MNIST

        self.train_set = MNIST(
            root="~/datasets", train=True, download=True, transform=self.transform
        )
        self.val_set = MNIST(root="~/datasets", train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
