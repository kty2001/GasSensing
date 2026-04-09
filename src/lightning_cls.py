import os
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  # 데이터 분석용
from datetime import datetime
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
)
from src.model import create_model # models.py에서 create_model 불러오기

class GasClsModel(L.LightningModule):
    def __init__(self, model_name, input_length, num_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 모델 생성
        self.model = create_model(model=model_name, input_length=input_length, output_dim=num_classes)
        
        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics (Val)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        
        # Metrics (Test)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # ⚡ 교수님 요청 분석용 리스트 (Confidence 저장)
        self.test_preds = []
        self.test_targets = []
        self.test_confidences = [] 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
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
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # ⚡ [핵심] Softmax로 확률(Confidence) 계산
        probs = torch.softmax(logits, dim=1) 
        confidence, preds = torch.max(probs, dim=1) # 가장 높은 확률값과 인덱스
        
        targets = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y)

        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        
        # 분석용 데이터 저장 (CPU로 이동)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.test_confidences.append(confidence.detach().cpu())

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def on_test_epoch_end(self):
        # 1. 현재 GPU가 가진 데이터 모으기
        if not self.test_preds: return
        
        local_preds = torch.cat(self.test_preds)
        local_targets = torch.cat(self.test_targets)
        local_confidences = torch.cat(self.test_confidences)
        
        # ⚡ 2. [DDP 핵심] 모든 GPU의 데이터를 하나로 수집 (all_gather)
        # self.all_gather는 DDP 환경에서만 작동하며, 단일 GPU일 땐 그냥 통과됩니다.
        all_preds = self.all_gather(local_preds)
        all_targets = self.all_gather(local_targets)
        all_confidences = self.all_gather(local_confidences)
        
        # all_gather를 하면 차원이 하나 늘어납니다 (GPU수, 데이터수). 다시 1차원으로 폅니다.
        if all_preds.dim() > 1:
            all_preds = all_preds.view(-1)
            all_targets = all_targets.view(-1)
            all_confidences = all_confidences.view(-1)

        # ⚡ 3. 대장 GPU(Rank 0)에서만 저장 및 시각화 실행
        # 이걸 안 하면 GPU 4개가 서로 저장하겠다고 싸웁니다.
        if self.trainer.is_global_zero:
            # 텐서를 넘파이로 변환 (CPU 이동)
            preds_np = all_preds.cpu().numpy()
            targets_np = all_targets.cpu().numpy()
            confidences_np = all_confidences.cpu().numpy()
            
            # --- (이하 저장 로직은 동일) ---
            os.makedirs("./result_analysis", exist_ok=True)
            timestamp = datetime.now().strftime('%m%d_%H%M%S')
            
            # (1) 데이터프레임 저장
            df = pd.DataFrame({
                'Target': targets_np,
                'Prediction': preds_np,
                'Confidence': confidences_np,
                'Correct': (targets_np == preds_np)
            })
            csv_path = os.path.join("./result_analysis", f"reliability_{self.hparams.model_name}_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n[Rank 0] 📄 전체 통합 데이터({len(df)}개) 저장됨: {csv_path}")

            # (2) 히스토그램
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='Confidence', hue='Correct', bins=20, multiple="stack", kde=True)
            plt.title(f"Confidence Distribution (Total: {len(df)})")
            hist_path = os.path.join("./result_analysis", f"conf_hist_{self.hparams.model_name}_{timestamp}.png")
            plt.savefig(hist_path, dpi=300)
            plt.close()

            # (3) Confusion Matrix
            cm = confusion_matrix(targets_np, preds_np)
            class_names = ["acetone", "benzene", "toluene"]
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.title(f"Confusion Matrix (Total: {len(df)})")
            cm_path = os.path.join("./result_analysis", f"cm_{self.hparams.model_name}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close()
            
        # 메모리 정리
        self.test_preds.clear()
        self.test_targets.clear()
        self.test_confidences.clear()

    # ⚡ [빠진 부분 추가] 이게 없어서 에러가 났었습니다!
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)