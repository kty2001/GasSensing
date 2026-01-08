import os
import glob
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.model import create_model
from src.dataset import GasDataModule
from src.utils import SEED


L.seed_everything(SEED)


class GasClsModel(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        input_length: int,
        num_classes: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(
            model=model_name,
            input_length=input_length,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def build_samples(df: dict, gas_to_label: dict):
    samples = []
    labels = []

    for gas, label in gas_to_label.items():
        merge = df[gas]["merge"].to_numpy()
        time = merge[:, 0]
        signal = merge[:, 1:]

        for i in range(signal.shape[1]):
            samples.append(signal[:, i].astype(np.float32))
            labels.append(label)

    x = np.stack(samples)
    y = np.array(labels)

    print("x shape", x.shape)
    print("y shape", y.shape)
    return x, y

def main(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath= f'{args.save}',
        filename= f'{args.model_name}-'+'{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15
    )
    wandb_logger = WandbLogger(project="Gas")
    
    gas_to_label = {
        "acetone": 0,
        "benzene": 1,
        "toluene": 2,
    }
    
    df = {}
    for path in glob.glob("data/pkl/*.pkl"):
        filename = os.path.splitext(os.path.basename(path))[0]
        gas, data_type = filename.split("_", 1)

        if gas not in df:
            df[gas] = {}

        with open(path, "rb") as f:
            obj = pickle.load(f)

        df[gas][data_type] = obj
    X, y = build_samples(df, gas_to_label)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    model = GasClsModel(args.model_name, input_length=7300)
    trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        max_epochs=args.epoch,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model, GasDataModule(train_data, args.batch))
    trainer.test(model, GasDataModule(test_data, args.batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='./data/pkl')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-mn', '--model_name', type=str, default='cnn1d')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('-dv', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-m', '--mode', type=str, default='train')
    args = parser.parse_args()
    
    main(args)
