from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam, SGD
from torch.tensor import Tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Network(LightningModule):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_outputs),
            nn.LogSoftmax(dim=1),
        )
        self.net.apply(self.init_weights)

        # Metrics
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    @staticmethod
    def init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        return  self.net(x)

    def configure_optimizers(self):
        # return Adam(self.parameters(), lr=1e-3)
        return SGD(self.parameters(), lr=0.1)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        accuracy = self.train_acc(logits, y)
        self.log('loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('accuracy', accuracy, on_step=False, on_epoch=True, logger=True)
        # Step automated by PL
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        return loss

    def training_epoch_end(self, tensors):
        items = super().get_progress_bar_dict()
        print(f"\n{items}\n")
        tensorboard = self.logger.experiment
        # tensorboard.add_image()
        for name, param in self.named_parameters():
            if 'bn' not in name:
                tensorboard.add_histogram(
                    f"Layer {name}",
                    param,
                    self.current_epoch
                )
                tensorboard.add_histogram(
                    f"Gradient {name}",
                    param.grad,
                    self.current_epoch
                )
        # tensorboard.add_figure(...)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        accuracy = self.valid_acc(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, logger=True)
        return {"loss": loss, "preds": y_hat, "target": y}

    def validation_epoch_end(self, outputs):
        N_CLASSES = 10
        tensorboard = self.logger.experiment

        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        confusion_matrix = pl.metrics.functional.confusion_matrix(
            preds,
            targets,
            num_classes=N_CLASSES,
            normalize="true",
        )
        df_cm = pd.DataFrame(
            confusion_matrix.numpy(),
            index=range(N_CLASSES),
            columns=range(N_CLASSES)
        )
        plt.figure(figsize = (10,7))
        fig = sns.heatmap(
            df_cm,
            annot=True,
            cmap='Blues',
        ).set(
            xlabel="Predictions",
            ylabel="Targets"
        ).get_figure()
        plt.close(fig)
        tensorboard.add_figure(
            "Confusion matrix",
            fig,
            self.current_epoch
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('test_loss', loss)


class FashionData(LightningDataModule):
    _N_WORKERS = 2

    def __init__(
        self,
        batch_size: int = 128,
        resize: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self._batch_size = batch_size
        self._resize = resize

        self.train_dataset = Dataset()
        self.val_dataset = Dataset()
        self.test_dataset = Dataset()

    def setup(self, stage):
        trans = [transforms.ToTensor()]
        if self._resize:
            trans.insert(0, transforms.Resize(self._resize))
        trans = transforms.Compose(trans)

        mnist_train = torchvision.datasets.FashionMNIST(
            root="../data",
            train=True,
            transform=trans,
            download=True
        )
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../data",
            train=False,
            transform=trans,
            download=True
        )
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._N_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._N_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._batch_size)


def main():

    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='max',
        period=5
    )

    data = FashionData()
    model = Network(784, 10)
    trainer = pl.Trainer(
        #gpus=0,
        max_epochs=5,
        check_val_every_n_epoch=1,
        #precision=16,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
