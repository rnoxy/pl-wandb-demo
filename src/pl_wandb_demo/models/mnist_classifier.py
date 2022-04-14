import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler


class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc1 = torch.nn.Linear(784, 10)

    def forward(self, x):
        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
