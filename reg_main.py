import os
import argparse
import glob
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split, KFold 
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.model import create_model
from src.dataset import GasDataModule
from src.utils import SEED
import torch
import torch.nn.functional as F

L.seed_everything(SEED)

# --------------------------------------------------------------------------------
# GasRegModel (수정 없음)
# --------------------------------------------------------------------------------
class GasRegModel(L.LightningModule):
    def __init__(self, model_name, input_length, output_dim=1, lr=1e-4, max_ppm=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model=model_name, input_length=input_length, output_dim=output_dim)

    def forward(self, x):
        return self.model(x)

    def _calculate_metrics(self, preds, y):
        preds, y = preds.view(-1), y.view(-1)
        loss = F.mse_loss(preds, y) 

        preds_ppm = preds * self.hparams.max_ppm
        y_ppm = y * self.hparams.max_ppm
        
        mask = y_ppm > 0.001
        if torch.sum(mask) > 0:
            diff = (y_ppm[mask] - preds_ppm[mask]) / y_ppm[mask]
            mape = torch.mean(torch.abs(diff)) * 100
            mspe = torch.mean(diff ** 2) * 100
        else:
            mape = torch.tensor(0.0, device=self.device)
            mspe = torch.tensor(0.0, device=self.device)
            
        return loss, mspe, mape 

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, mspe, mape = self._calculate_metrics(self(x), y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mape", mape, on_epoch=True, prog_bar=True)
        self.log("train_mspe", mspe, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, mspe, mape = self._calculate_metrics(self(x), y.float())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        self.log("val_mspe", mspe, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, mspe, mape = self._calculate_metrics(self(x), y.float())
        # Test 결과 로깅
        self.log("test_loss", loss)
        self.log("test_mape", mape)
        self.log("test_mspe", mspe)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# --------------------------------------------------------------------------------
# 데이터 로드 함수
# --------------------------------------------------------------------------------
def load_target_gas_data(target_gas: str, data_dir: str):
    print(f"📂 [{target_gas}] 데이터 로드 시작...")
    
    label_path = os.path.join(data_dir, "ppm_label_renew", f"{target_gas}_label_ppm.csv")
    try:
        y_data = np.loadtxt(label_path, delimiter=',', skiprows=1)
        y = y_data if y_data.ndim == 1 else y_data[:, 0]
    except Exception as e:
        print(f"❌ 라벨 로드 실패: {e}")
        return None, None

    pkl_path = os.path.join(data_dir, "pickle", f"{target_gas}_merge.pkl")
    if os.path.exists(pkl_path):
        df_raw = pd.read_pickle(pkl_path)
        data_numpy = df_raw.to_numpy() if hasattr(df_raw, "to_numpy") else np.array(df_raw)
        x = data_numpy[:, 1:].T.astype(np.float32) 
    else:
        print(f"❌ 스펙트럼 파일 없음: {pkl_path}")
        return None, None

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    mask = y > 0
    x = x[mask]
    y = y[mask]
    
    print(f"✅ 데이터 로드 완료: X shape={x.shape}, y shape={y.shape} (0ppm 제거됨)")
    return x, y


# --------------------------------------------------------------------------------
# Main 함수
# --------------------------------------------------------------------------------
def main(args):
    if not os.path.exists(args.save): os.makedirs(args.save)

    # 1. 데이터 로드
    X, y = load_target_gas_data(args.target_gas, "./data")
    if X is None: return

    # 2. 전처리
    y_max_val = np.max(y)
    y_scaled = y / y_max_val
    print(f"⚖️ Max PPM: {y_max_val} (Scaled 0~1)")

    # 3. Test Set 분리 (Hold-out)
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y_scaled, 
        test_size=0.2,     
        random_state=SEED, 
        shuffle=True
    )
    
    final_test_dataset = list(zip(X_test_final, y_test_final))
    
    # 4. K-Fold 설정
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # ⚡ [그룹핑 핵심] 이 이름으로 5개의 Fold와 1개의 Test가 묶입니다.
    # 예: "acetone_cnn1d_Exp" 라는 그룹 아래에 6개의 그래프가 생김
    group_name = f"{args.target_gas}_{args.model_name}_KFold_Exp"
    
    print(f"\n🚀 K-Fold 학습 시작 (5-Fold, Train:Val = 80:20)")
    
    best_fold_score = float('inf')
    best_fold_idx = 0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
        print(f"\n🔄 [Fold {fold+1}/5] 학습 진행 중...")
        
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
        
        train_dataset = list(zip(X_train, y_train))
        val_dataset = list(zip(X_val, y_val))
        
        data_module = GasDataModule(train_data=train_dataset, val_data=val_dataset, batch_size=args.batch)
        
        model = GasRegModel(
            args.model_name, 
            input_length=X.shape[1], 
            max_ppm=y_max_val, 
            lr=1e-4
        )
        
        wandb_logger = WandbLogger(
            project="Gas-Reg-KFold", 
            name=f"{args.target_gas}_Fold{fold+1}",
            group=group_name, # 그룹 지정
            reinit=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss', mode='min', 
            dirpath=args.save, 
            filename=f'{args.model_name}-Fold{fold+1}-{{val_loss:.4f}}',
            save_top_k=1
        )
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)
        
        trainer = L.Trainer(
            accelerator=args.device, devices=1, max_epochs=args.epoch,
            logger=wandb_logger, callbacks=[checkpoint_callback, early_stopping],
            num_sanity_val_steps=0
        )
        
        trainer.fit(model, data_module)
        val_result = trainer.validate(model, data_module)[0]
        
        if val_result['val_loss'] < best_fold_score:
            best_fold_score = val_result['val_loss']
            best_fold_idx = fold + 1
        
        wandb.finish() 

    # -------------------------------------------------------------------------
    # 5. [수정됨] 최종 Test Set 평가 및 WandB 로깅
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"🏆 Best Fold: Fold {best_fold_idx} (Val Loss: {best_fold_score:.5f})")
    print(f"🧪 Final Test Set 평가 시작...")
    
    # Best Checkpoint 찾기
    best_ckpt_path = glob.glob(os.path.join(args.save, f"{args.model_name}-Fold{best_fold_idx}-*.ckpt"))[0]
    
    # ⚡ weights_only=False 추가 (에러 해결)
    best_model = GasRegModel.load_from_checkpoint(
        best_ckpt_path, 
        model_name=args.model_name, 
        input_length=X.shape[1], 
        max_ppm=y_max_val,
        weights_only=False 
    )
    
    # ⚡ [NEW] Test 전용 WandbLogger 생성
    # group=group_name을 그대로 사용하여 K-Fold 그래프들과 함께 묶이게 함
    test_logger = WandbLogger(
        project="Gas-Reg-KFold",
        name=f"{args.target_gas}_Final_Test",
        group=group_name,  # 동일 그룹 사용
        job_type="test",
        reinit=True
    )

    # Test Trainer (Logger 연결)
    test_trainer = L.Trainer(
        accelerator=args.device, 
        devices=1, 
        logger=test_logger # False 대신 logger 사용
    )
    
    test_trainer.test(best_model, GasDataModule(test_data=final_test_dataset, batch_size=args.batch))
    
    # Test Run 종료
    wandb.finish()
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='./data/pkl')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-mn', '--model_name', type=str, default='cnn1d')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('-dv', '--device', type=str, default='gpu')
    parser.add_argument('-tg', '--target_gas', type=str, default='benzene')
    
    args = parser.parse_args()
    main(args)