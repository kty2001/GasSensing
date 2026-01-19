import os
import glob
import argparse
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from src.model import create_model
from src.dataset import GasDataModule
from src.utils import SEED, build_samples

warnings.filterwarnings("ignore")
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

        # One-Hot Encoding loss_fn
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_acc = MulticlassAccuracy(num_classes=3)
        self.val_precision = MulticlassPrecision(num_classes=3, average="macro")
        self.val_recall = MulticlassRecall(num_classes=3, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=3, average="macro")
        
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # # Index
        # loss = F.cross_entropy(logits, y)
        # acc = (logits.argmax(dim=1) == y).float().mean()

        # One-Hot Encoding
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        acc = (preds == targets).float().mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # # Index
        # loss = F.cross_entropy(logits, y)
        # acc = (logits.argmax(dim=1) == y).float().mean()

        # One-Hot Encoding
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        acc = (preds == targets).float().mean()

        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)
        self.log("val_f1", self.val_f1)

    def on_test_epoch_start(self):
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

        self.test_preds.clear()
        self.test_targets.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # # Index
        # preds = torch.argmax(logits, dim=1)
        # loss = F.cross_entropy(logits, y)
        # acc = (logits.argmax(dim=1) == y).float().mean()

        # One-Hot Encoding
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)

        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log("test_loss", loss)
        self.log("test/acc", self.test_acc, prog_bar=True)
        self.log("test/precision", self.test_precision)
        self.log("test/recall", self.test_recall)
        self.log("test/f1", self.test_f1)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()
        cm = confusion_matrix(targets, preds)

        class_names = ["acetone", "benzene", "toluene"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")

        save_path = os.path.join("./result", f"cm_{args.model_name}_{datetime.now().strftime("%m%d_%H%M%S")}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        # reset (안 하면 여러 test run 시 누적됨)
        self.test_preds.clear()
        self.test_targets.clear()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main(args):
    os.makedirs(args.save, exist_ok=True)
    
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
    if args.data == 'pkl':
        wandb_logger = WandbLogger(
            project="Gas",
            name=f"{args.model_name}_{datetime.now().strftime("%m%d_%H%M%S")}"
        )
    elif args.data == 'del':
        wandb_logger = WandbLogger(
            project="Gas",
            name=f"filtered_{args.model_name}_{datetime.now().strftime("%m%d_%H%M%S")}"
    )
    else:
        raise ValueError(f"Unknown data type: {args.data}")
    
    wandb_run = wandb_logger.experiment
    fold_table = wandb.Table(columns=["Fold", "Acc", "Precision", "Recall", "F1"])

    X, y_index, y_onehot = build_samples(args.data)

    X_train, X_test, y_index_train, y_index_test, y_onehot_train, y_onehot_test = \
        train_test_split(
            X, y_index, y_onehot,
            test_size=0.2,
            random_state=SEED,
            stratify=y_index
        )

    train_data = (X_train, y_index_train, y_onehot_train)
    test_data = (X_test, y_index_test, y_onehot_test)

    print("Start Train")
    if args.mode == 'train':
        print("---------------K-Fold---------------")
        
        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )

        fold_results = {
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_index_train)):
            print(f"\n--- Fold {fold+1}/5 ---")

            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_onehot_train[tr_idx], y_onehot_train[val_idx]

            fold_dm = GasDataModule(
                train_data=(X_tr, y_tr),
                val_data=(X_val, y_val),
                batch_size=args.batch,
            )

            fold_model = GasClsModel(
                model_name=args.model_name,
                input_length=7300
            )

            fold_trainer = L.Trainer(
                accelerator=args.device,
                devices=1,
                max_epochs=args.epoch,
                logger=wandb_logger,
                enable_checkpointing=False
            )

            fold_trainer.fit(fold_model, datamodule=fold_dm)

            val_metrics = fold_trainer.callback_metrics

            val_acc = val_metrics["val_acc"].item()
            val_precision = val_metrics["val_precision"].item()
            val_recall = val_metrics["val_recall"].item()
            val_f1 = val_metrics["val_f1"].item()

            fold_results["acc"].append(val_acc)
            fold_results["precision"].append(val_precision)
            fold_results["recall"].append(val_recall)
            fold_results["f1"].append(val_f1)

            fold_table.add_data(
                fold + 1,
                val_acc,
                val_precision,
                val_recall,
                val_f1
            )

        print("\n[Per-Fold Validation Result]")
        print(f"{'Fold':<6} {'Acc':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
        for i in range(len(fold_results["acc"])):
            print(
                f"{i+1:<6} "
                f"{fold_results['acc'][i]:>12.8f} "
                f"{fold_results['precision'][i]:>12.8f} "
                f"{fold_results['recall'][i]:>12.8f} "
                f"{fold_results['f1'][i]:>12.8f}"
            )
        wandb_run.log({"per_fold_table": fold_table})

        print("\n[K-Fold Validation Result]")
        for k, v in fold_results.items():
            print(
                f"{k.upper():<10} "
                f"Mean = {np.mean(v):.8f}, "
                f"Std = {np.std(v):.8f}"
            )
        wandb_run.log({
            "kfold/acc_mean": np.mean(fold_results["acc"]),
            "kfold/acc_std": np.std(fold_results["acc"]),
            "kfold/precision_mean": np.mean(fold_results["precision"]),
            "kfold/precision_std": np.std(fold_results["precision"]),
            "kfold/recall_mean": np.mean(fold_results["recall"]),
            "kfold/recall_std": np.std(fold_results["recall"]),
            "kfold/f1_mean": np.mean(fold_results["f1"]),
            "kfold/f1_std": np.std(fold_results["f1"]),
        })

        print("---------------Final---------------")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_onehot_train,
            test_size=0.1,
            random_state=SEED,
            stratify=y_index_train
        )

        final_dm = GasDataModule(
            train_data=(X_tr, y_tr),
            val_data=(X_val, y_val),
            test_data=(X_test, y_onehot_test),
            batch_size=args.batch,
        )

        final_model = GasClsModel(args.model_name, input_length=7300)

        final_trainer = L.Trainer(
            accelerator=args.device,
            devices=1,
            max_epochs=args.epoch,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
        )

        final_trainer.fit(final_model, datamodule=final_dm)
        final_trainer.test(final_model, datamodule=final_dm)

    elif args.mode == 'test':
        test_dm = GasDataModule(
            train_data=None,
            val_data=None,
            test_data=(X_test, y_onehot_test),
            batch_size=args.batch,
        )

        test_model = GasClsModel.load_from_checkpoint(
            args.ckpt,
            model=create_model(
                model=args.model_name,
                input_length=7300,
                num_classes=3,
            ),
            num_classes=3,
        )

        test_trainer = L.Trainer(
            accelerator=args.device,
            devices=1,
            max_epochs=args.epoch,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
        )

        test_trainer.test(test_model, datamodule=test_dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='pkl')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    parser.add_argument('-mn', '--model_name', type=str, default='mlp')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=64)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('-dv', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-m', '--mode', type=str, default='train')
    args = parser.parse_args()
    
    main(args)
