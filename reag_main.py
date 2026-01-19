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


class GasRegModel(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        input_length: int,
        output_dim: int = 1,
        lr: float = 1e-4,
        max_ppm: float = 1.0, 
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(
            model=model_name,
            input_length=input_length,
            output_dim=output_dim,
        )

    def forward(self, x):
        return self.model(x)

    def _calculate_metrics(self, preds, y):
        """Loss, MSPE, MAPE만 계산"""
        # 1. 차원 맞추기
        preds = preds.view(-1)
        y = y.view(-1)

        # 2. 학습용 Loss (Scaled 0~1) - 학습은 이걸로 함
        loss = F.mse_loss(preds, y)

        # 3. 지표용 스케일 복원 (PPM 단위)
        # ⚠️ max_ppm이 1.0이면 복원이 안 됩니다. main.py에서 꼭 값을 넘겨주세요.
        preds_ppm = preds * self.hparams.max_ppm
        y_ppm = y * self.hparams.max_ppm
        
        # 4. MSPE & MAPE 계산 (0이 아닌 구간만)
        # [수정] 1.0 -> 0.001로 기준을 낮췄습니다. (스케일링 문제 방지)
        mask = y_ppm > 0.001 
        
        if torch.sum(mask) > 0:
            diff = (y_ppm[mask] - preds_ppm[mask]) / y_ppm[mask]
            
            # MAPE: (|y-p|/y) * 100
            mape = torch.mean(torch.abs(diff)) * 100
            
            # MSPE: ((y-p)/y)^2 * 100
            mspe = torch.mean(diff ** 2) * 100
        else:
            # 가스가 없는 배치거나, 복원 값이 너무 작으면 0 반환
            mape = torch.tensor(0.0, device=self.device)
            mspe = torch.tensor(0.0, device=self.device)
        
        return loss, mspe, mape

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) 

        # [수정] 3개만 반환
        loss, mspe, mape = self._calculate_metrics(logits, y.float())

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mape", mape, on_epoch=True, prog_bar=True)
        self.log("train_mspe", mspe, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss, mspe, mape = self._calculate_metrics(logits, y.float())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        self.log("val_mspe", mspe, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss, mspe, mape = self._calculate_metrics(logits, y.float())

        self.log("test_loss", loss)
        self.log("test_mape", mape)
        self.log("test_mspe", mspe)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def load_target_gas_data(target_gas: str, data_dir: str):
    print(f"📂 [{target_gas}] 데이터 로드 시작...")
    
    # CSV 라벨 로드
    label_path = os.path.join(data_dir, "ppm_label_renew", f"{target_gas}_label_ppm.csv")
    try:
        y_data = np.loadtxt(label_path, delimiter=',', skiprows=1)
        y = y_data if y_data.ndim == 1 else y_data[:, 0]
    except Exception as e:
        print(f"❌ 라벨 로드 실패 ({label_path}): {e}")
        return None, None

    # Pickle 스펙트럼 로드
    pkl_path = os.path.join(data_dir, "pickle", f"{target_gas}_merge.pkl")
    if os.path.exists(pkl_path):
        df_raw = pd.read_pickle(pkl_path)
        if hasattr(df_raw, "to_numpy"):
            data_numpy = df_raw.to_numpy()
        else:
            data_numpy = np.array(df_raw)
        
        # 전처리: 헤더(1행), 시간(1열) 제거 및 전치
        x = data_numpy[:, 1:].T.astype(np.float32)
    else:
        print(f"❌ 스펙트럼 파일 없음: {pkl_path}")
        return None, None

    # 데이터 개수 동기화
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    mask = y > 0  # 0보다 큰 것만 선택
    x = x[mask]
    y = y[mask]
    
    print(f"🧹 0ppm 제거 완료: {len(y)}개 남음")

    print(f"✅ 데이터 로드 완료: X shape={x.shape}, y shape={y.shape}")
    return x, y

def main(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    wandb_logger = WandbLogger(project="Gas-Regression", name=f"{args.target_gas}_{args.model_name}")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath= f'{args.save}',
        filename= f'{args.model_name}-{args.target_gas}-'+'{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15
    )
    
    X, y = load_target_gas_data(args.target_gas, "./data")

    if X is None: return

    # stratify 제거 및 shuffle 옵션
    
    y_max_val = np.max(y)
    
    y_scaled = y / y_max_val
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled,
        test_size=0.2,
        random_state=SEED,
        shuffle=True, 
    )

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    input_len = X.shape[1]
    model = GasRegModel(args.model_name,
                        input_length=input_len,
                        max_ppm=y_max_val)
    
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
    parser.add_argument('-tg', '--target_gas', type=str, default='benzene', help='Target gas name')
    
    args = parser.parse_args()
    
    main(args)