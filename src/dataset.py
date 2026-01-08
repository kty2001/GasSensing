import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning as L

from src.utils import SEED


L.seed_everything(SEED)


class GasDataModule(L.LightningDataModule):
    def __init__(
            self,
            data: tuple,
            batch_size: int = 64,
        ):
        super().__init__()

        self.x, self.y = data
        self.batch_size = batch_size

    def setup(self, stage: str):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.batch_size

        if stage == 'fit':
            train_x, val_x, train_y, val_y = train_test_split(
                self.x, self.y,
                test_size=0.2,
                random_state=SEED,
                stratify=self.y,
            )
            self.train_dataset = list(zip(train_x, train_y))
            self.val_dataset = list(zip(val_x, val_y))
            
        elif stage == 'test':
            self.test_dataset = list(zip(self.x, self.y))

        elif stage == 'predict':
            self.pred_dataset = 1
            pass

    def _collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.tensor(np.stack(x), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        return x, y
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_per_device, shuffle=True, collate_fn=self._collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._collate_fn)
    