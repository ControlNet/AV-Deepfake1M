import timm

from lightning.pytorch import LightningModule
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam


class Xception(LightningModule):
    def __init__(self, lr, distributed=False):
        super(Xception, self).__init__()
        self.lr = lr
        self.model = timm.create_model('xception', pretrained=True, num_classes=1)
        self.loss_fn = BCEWithLogitsLoss()
        self.distributed = distributed

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return [optimizer]
