import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 10),
            nn.ReLU())
    
    def forward(self, x):
        return self.classifier(x.view(x.size(0), -1))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x.view(x.size(0), -1))
        return nn.functional.cross_entropy(y_hat, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x.view(x.size(0), -1))
        self.log_dict({'accuracy': FM.accuracy(torch.argmax(y_hat, dim=1), y)}) 

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)


train_loader = DataLoader(MNIST('./', train=True, download=True, transform=transforms.ToTensor()), batch_size=1024, num_workers=12)
test_loader = DataLoader(MNIST('./', train=False, download=True, transform=transforms.ToTensor()), batch_size=1024, num_workers=12)

trainer = pl.Trainer(max_epochs=10, gpus=1)
model = Classifier()

trainer.fit(model, train_loader)
trainer.test(model, test_dataloaders=test_loader)
