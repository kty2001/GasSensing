import os
import glob
import argparse
import warnings
import shutil
import numpy as np
import pandas as pd
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# PyTorch & Lightning
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader

# Custom Modules
from src.dataset import GasDataModule
from src.utils import SEED, build_samples, get_next_run_dir
from src.lightning_reg import GasRegModel
from src.lightning_cls import GasClsModel

warnings.filterwarnings("ignore")
L.seed_everything(SEED)

# ==============================================================================
# [1] 회귀 (Regression) 실행 로직
# ==============================================================================
def run_regression(args):
    # ⚡ [수정 1] 실행 폴더 생성 (runs, runs1, runs2...)
    # 주의: Multi-GPU(DDP)에서는 Rank 0만 생성하도록 하면 좋지만, 
    # 간단하게 여기서는 동시에 체크해서 같은 폴더를 쓰도록 유도합니다.
    # (실제로는 timestamp 기반이 DDP에선 더 안전하지만 요청하신 runs 구조를 따릅니다.)
    save_dir = get_next_run_dir("runs", parent_dir=args.save)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"📂 이번 실험 저장 경로: {save_dir}")

    # 데이터 로드 (기존과 동일)
    def load_data(target_gas, data_dir):
        print(f"📂 [{target_gas}] 데이터 로드 시작...")
        label_path = os.path.join(data_dir, "ppm_label_renew", f"{target_gas}_label_ppm.csv")
        pkl_path = os.path.join(data_dir, "pickle", f"{target_gas}_merge.pkl")
        
        if not os.path.exists(pkl_path) or not os.path.exists(label_path):
            print("❌ 데이터 파일이 없습니다.")
            return None, None
            
        y_data = np.loadtxt(label_path, delimiter=',', skiprows=1)
        y = y_data if y_data.ndim == 1 else y_data[:, 0]
        
        df_raw = pd.read_pickle(pkl_path)
        data_numpy = df_raw.to_numpy() if hasattr(df_raw, "to_numpy") else np.array(df_raw)
        x = data_numpy[:, 1:].T.astype(np.float32)

        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        mask = y > 0
        return x[mask], y[mask]

    X, y = load_data(args.target_gas, "./data")
    if X is None: return

    y_max_val = np.max(y)
    y_scaled = y / y_max_val
    print(f"⚖️ Max PPM: {y_max_val} (Scaled 0~1)")

    # Split
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y_scaled, test_size=0.2, random_state=SEED, shuffle=True
    )
    final_test_dataset = list(zip(X_test_final, y_test_final))

    # K-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    group_name = f"{args.target_gas}_{args.model_name}_REG_KFold"

    best_fold_score = float('inf')
    best_fold_idx = 0

    print(f"\n🚀 [Regression] K-Fold 시작 (Gas: {args.target_gas})")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
        print(f"\n🔄 [Fold {fold+1}/5] ...")
        
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

        dm = GasDataModule(
            train_data=list(zip(X_tr, y_tr)), 
            val_data=list(zip(X_val, y_val)), 
            batch_size=args.batch, 
            task='reg'
        )
        model = GasRegModel(args.model_name, input_length=X.shape[1], max_ppm=y_max_val)

        logger = WandbLogger(project="Gas-Integrated-Final", name=f"{args.target_gas}_Fold{fold+1}", group=group_name, reinit=True)
        
        # ⚡ [수정 2] 체크포인트 이름에 '가스이름' 추가
        # 예: benzene_cnn1d-Fold1-val_loss=0.0123.ckpt
        ckpt_filename = f"{args.target_gas}_{args.model_name}-Fold{fold+1}-{{val_loss:.4f}}"
        
        ckpt = ModelCheckpoint(
            monitor='val_loss', 
            mode='min', 
            dirpath=save_dir,  # ⚡ 위에서 만든 runs 폴더 사용
            filename=ckpt_filename, 
            save_top_k=1
        )
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)

        trainer = L.Trainer(
            accelerator='gpu' if args.device == 'gpu' else 'cpu',
            devices=args.gpus, 
            strategy='ddp',
            max_epochs=args.epoch,
            logger=logger,
            callbacks=[ckpt, early_stopping],
        )
        trainer.fit(model, dm)
        
        score = trainer.validate(model, dm)[0]['val_loss']
        if score < best_fold_score:
            best_fold_score = score
            best_fold_idx = fold + 1
        
        wandb.finish()

    # Final Test
    print(f"\n🏆 Best Fold: {best_fold_idx} (Loss: {best_fold_score:.5f})")
    
    # ⚡ [수정 3] Best 모델 찾아서 'Best_가스명.ckpt'로 복사
    # 저장된 파일 패턴: [target_gas]_[model_name]-Fold[best_idx]-*.ckpt
    pattern = os.path.join(save_dir, f"{args.target_gas}_{args.model_name}-Fold{best_fold_idx}-*.ckpt")
    found_ckpts = glob.glob(pattern)
    
    if found_ckpts:
        best_ckpt_path = found_ckpts[0]
        best_save_name = os.path.join(save_dir, f"Best_{args.target_gas}.ckpt")
        shutil.copy(best_ckpt_path, best_save_name)
        print(f"💾 Best Model Copied to: {best_save_name}")

        # [수정] 1. 테스트 전용 Logger
        test_logger = WandbLogger(project="Gas-Integrated-Final", name=f"{args.target_gas}_Test", group=group_name, job_type="test", reinit=True)
        
        # [수정] 2. 테스트 전용 Trainer (DDP 유지)
        test_trainer = L.Trainer(
            accelerator='gpu' if args.device == 'gpu' else 'cpu',
            devices=args.gpus,       # ⚡ 4개 유지
            strategy='ddp',          # ⚡ DDP 유지
            logger=test_logger
        )

        test_dm = GasDataModule(test_data=final_test_dataset, batch_size=args.batch, task='reg')
        
        print(f"🧪 Final Test with Best Checkpoint: {best_save_name}")
        
        best_model = GasRegModel.load_from_checkpoint(
            best_save_name, 
            model_name=args.model_name, 
            input_length=X.shape[1], 
            weights_only=False 
        )
        
        # ckpt_path=None (생략)으로 설정
        test_trainer.test(best_model, datamodule=test_dm)
        
        wandb.finish()
    else:
        print("❌ Best Checkpoint 파일을 찾을 수 없습니다.")
        return


# ==============================================================================
# [2] 분류 (Classification) 실행 로직
# ==============================================================================
def run_classification(args):
    # ⚡ [수정 1] 실행 폴더 생성
    save_dir = get_next_run_dir("runs", parent_dir=args.save)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"📂 이번 실험 저장 경로: {save_dir}")

    print(f"\n🚀 [Classification] 데이터 로드 및 전처리 (Type: {args.data_type})")
    
    X, y_index, y_onehot = build_samples(args.data_type)

    X_train, X_test, y_idx_train, y_idx_test, y_hot_train, y_hot_test = train_test_split(
        X, y_index, y_onehot, test_size=0.2, random_state=SEED, stratify=y_index
    )

    group_name = f"{args.model_name}_CLS_KFold_{datetime.now().strftime('%m%d_%H%M')}"
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    best_loss = float('inf')
    best_ckpt_path = None
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_idx_train)):
        print(f"\n🔄 [Fold {fold+1}/5] ...")
        
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_hot_train[tr_idx], y_hot_train[val_idx]

        dm = GasDataModule(
            train_data=(X_tr, y_tr), 
            val_data=(X_val, y_val), 
            batch_size=args.batch, 
            task='cls'
        )
        
        model = GasClsModel(args.model_name, input_length=X.shape[1], num_classes=3)
        
        logger_name = f"Cls_{args.model_name}_Fold{fold+1}"
        logger = WandbLogger(project="Gas-Integrated-Final", name=logger_name, group=group_name, reinit=True)
        
        # ⚡ [수정 2] 체크포인트 이름: CLS_모델명...
        ckpt_filename = f"CLS_{args.model_name}-Fold{fold+1}-{{val_loss:.4f}}"
        
        ckpt = ModelCheckpoint(
            monitor='val_loss', 
            mode='min', 
            dirpath=save_dir,  # runs 폴더 사용
            filename=ckpt_filename, 
            save_top_k=1
        )
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)
        
        trainer = L.Trainer(
            accelerator='gpu' if args.device == 'gpu' else 'cpu',
            devices=args.gpus, 
            strategy='ddp',
            max_epochs=args.epoch,
            logger=logger,
            callbacks=[ckpt, early_stopping],
        )
        trainer.fit(model, dm)
        
        score = ckpt.best_model_score.item()
        if score < best_loss:
            best_loss = score
            best_ckpt_path = ckpt.best_model_path
        
        wandb.finish()

    # Final Test
    print(f"\n🏆 Best Val Loss: {best_loss:.5f}")
    
    # ⚡ [수정 3] Best 모델 복사
    if best_ckpt_path:
        best_save_name = os.path.join(save_dir, "Best_CLS.ckpt")
        shutil.copy(best_ckpt_path, best_save_name)
        print(f"💾 Best Model Copied to: {best_save_name}")
    
        # [수정] 1. 테스트 전용 Logger 생성
        test_logger = WandbLogger(project="Gas-Integrated-Final", name=f"Cls_{args.model_name}_Test", group=group_name, reinit=True)
        
        # [수정] 2. 테스트 전용 Trainer 생성 (중요: DDP 설정 유지!)
        # 기존 trainer를 재사용하면 이미 닫힌 wandb run을 쓰려다가 에러가 납니다.
        # 따라서 새 trainer를 만들되, 현재 프로세스 그룹(DDP)과 맞게 설정을 동일하게 줍니다.
        test_trainer = L.Trainer(
            accelerator='gpu' if args.device == 'gpu' else 'cpu',
            devices=args.gpus,      # ⚡ 4개 유지 (중요)
            strategy='ddp',         # ⚡ DDP 유지 (중요)
            logger=test_logger      # 새 로거 연결
        )
        
        test_dm = GasDataModule(test_data=(X_test, y_hot_test), batch_size=args.batch, task='cls')
        
        print(f"🧪 Final Test with Best Checkpoint: {best_save_name}")
        
        # [수정] 3. 새로 만든 trainer로 테스트
        best_model = GasClsModel.load_from_checkpoint(
            best_save_name, 
            model_name=args.model_name, 
            input_length=X.shape[1], 
            num_classes=3,
            weights_only=False
        )
        
        # 로드된 모델로 테스트
        test_trainer.test(best_model, datamodule=test_dm)
        
        wandb.finish()
    
    
# ==============================================================================
# [3] 통합 파이프라인 (Pipeline: CLS -> REG) 실행 로직
# ==============================================================================
def run_pipeline(args):
    print(f"\n🚀 [Pipeline] 통합 추론 시작 (Classification -> Regression)")

    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    print(f"⚙️ Device set to: {device}")
    
    # ---------------------------------------------------------
    # 1. 모델 로드
    # ---------------------------------------------------------
    print("📥 모델 로드 중...")
    
    if not args.ckpt_cls:
        raise ValueError("❌ 분류 모델 체크포인트(--ckpt_cls)가 필요합니다.")
    
    # [분류 모델]
    cls_model = GasClsModel.load_from_checkpoint(
        args.ckpt_cls, model_name=args.model_name_cls, input_length=7300, num_classes=3, weights_only=False
    ).to(device)
    cls_model.eval()
    
    # [회귀 모델]
    reg_models = {}
    target_gases = ["acetone", "benzene", "toluene"] 
    
    for i, gas in enumerate(target_gases):
        ckpt_path = os.path.join(args.ckpt_dir, f"Best_{gas}.ckpt") 
        if not os.path.exists(ckpt_path):
             print(f"⚠️ 경고: {gas} 회귀 모델 파일이 없습니다 ({ckpt_path}).")
             continue
        
        # ⚡ [수정 1] max_ppm=1.0 인자 제거! 
        # (체크포인트에 저장된 원래 max_ppm 값을 쓰도록 함)
        reg_model = GasRegModel.load_from_checkpoint(
            ckpt_path, model_name=args.model_name_reg, input_length=7300, weights_only=False 
        ).to(device)
        
        reg_model.eval()
        reg_models[i] = reg_model
        
        # 디버깅: 로드된 max_ppm 확인
        print(f"   🔹 [{gas}] Model Loaded | Max PPM: {reg_model.hparams.max_ppm}")
        
    print("✅ 모든 모델 로드 완료!")

    # ---------------------------------------------------------
    # 2. 데이터 및 PPM 라벨 로드 (동기화)
    # ---------------------------------------------------------
    print("📂 데이터 및 PPM 라벨 로드 중...")
    
    # (1) 데이터 로드 (X, y_index) - build_samples는 보통 0ppm이 제거된 데이터를 반환한다고 가정
    X, y_index, _ = build_samples(args.data_type)
    
    # (2) PPM 라벨 로드
    y_ppm_list = []
    for gas in target_gases:
        label_path = f"./data/ppm_label_renew/{gas}_label_ppm.csv"
        try:
            ppm_data = np.loadtxt(label_path, delimiter=',', skiprows=1)
            ppm = ppm_data if ppm_data.ndim == 1 else ppm_data[:, 0]
            
            # ⚡ [수정 2] 0ppm 제거 필터 적용! (X 데이터와 줄 맞추기)
            # reg_main.py와 동일하게 0보다 큰 값만 남깁니다.
            ppm = ppm[ppm > 0]
            
            y_ppm_list.append(ppm)
        except Exception as e:
            print(f"❌ PPM 로드 실패 ({gas}): {e}")
            return

    # 전체 PPM 합치기
    y_ppm_all = np.concatenate(y_ppm_list)
    
    # 🚨 데이터 개수 검증 (매우 중요)
    if len(X) != len(y_ppm_all):
        print(f"\n⚠️ [CRITICAL WARNING] 데이터 개수 불일치 발생!")
        print(f"   - X (Input) count: {len(X)}")
        print(f"   - y (PPM) count  : {len(y_ppm_all)}")
        print("   -> 데이터 정렬이 어긋나서 회귀 성능이 망가질 수 있습니다.")
        print("   -> utils.py의 build_samples가 데이터를 어떻게 자르는지 확인이 필요합니다.")
        # 강제로 길이를 맞춰서 진행 (임시 조치)
        min_len = min(len(X), len(y_ppm_all))
        X = X[:min_len]
        y_index = y_index[:min_len]
        y_ppm_all = y_ppm_all[:min_len]
    
    # (3) Split (똑같은 시드로 섞어서 Test셋 분리)
    _, X_test, _, y_idx_test, _, y_ppm_test = train_test_split(
        X, y_index, y_ppm_all, test_size=0.2, random_state=SEED, stratify=y_index
    )
    
    # DataModule
    dm = GasDataModule(test_data=(X_test, y_idx_test), batch_size=args.batch, task='cls')
    dm.setup(stage='test')
    dataloader = dm.test_dataloader()

    # ---------------------------------------------------------
    # 3. 추론 루프
    # ---------------------------------------------------------
    total = 0
    cls_correct = 0
    
    true_ppms = []
    pred_ppms = []
    
    # ⚡ [분석용] 신뢰도 저장을 위한 리스트 추가
    all_confidences = []     
    low_conf_count = 0       # 신뢰도가 낮은 케이스 카운트
    CONF_THRESHOLD = 0.9     # (예시) 60% 미만이면 '불확실'로 간주
    
    print("\n🔍 추론 진행 중... (신뢰도 분석 포함)")
    
    current_idx = 0 
    
    results_list = []

    with torch.no_grad():
        for batch in dataloader:
            x, y_cls_true = batch 
            x = x.to(device)
            y_cls_true = y_cls_true.to(device)
            
            # [1] 분류 모델 예측
            cls_logits = cls_model(x)
            
            # 🔥 [핵심 추가] Logits -> Softmax -> 확률(Confidence) 변환
            cls_probs = torch.softmax(cls_logits, dim=1)        # (Batch, 3) 확률값
            cls_confidences, cls_preds = torch.max(cls_probs, dim=1) # 최대 확률(신뢰도)과 인덱스
            
            batch_size = x.size(0)
            
            # 정답 PPM (회귀 평가용)
            batch_true_ppms = y_ppm_test[current_idx : current_idx + batch_size]
            current_idx += batch_size

            for k in range(batch_size):
                total += 1
                curr_x = x[k].unsqueeze(0)
                
                # 예측값과 신뢰도 추출
                pred_idx = cls_preds[k].item()
                conf_val = cls_confidences[k].item() # 0.0 ~ 1.0 사이 값
                
                # 리스트에 저장 (나중에 평균 보려고)
                all_confidences.append(conf_val)

                # 정답 클래스 (One-Hot or Index)
                true_idx = int(torch.argmax(y_cls_true[k]).item() if y_cls_true.dim() > 1 else y_cls_true[k].item())
                
                # -----------------------------------------------------
                # ⚡ [교수님 요청] 실시간 신뢰도 모니터링 & 경고
                # -----------------------------------------------------
                # 신뢰도가 너무 낮으면 로그를 찍어봅니다.
                if conf_val < CONF_THRESHOLD:
                    low_conf_count += 1
                    print(f"⚠️ [Low Conf] Sample {total}: Pred={pred_idx}, True={true_idx}, Conf={conf_val:.4f} (불확실!)")

                # 인덱스 방어
                if k < len(batch_true_ppms):
                    true_ppm_val = batch_true_ppms[k]
                else:
                    break 

                # [분류 정확도 체크]
                if pred_idx == true_idx:
                    cls_correct += 1
                    
                    # [2] 회귀 (분류 성공 시)
                    if pred_idx in reg_models:
                        
                        # ⚡ (선택사항) 만약 신뢰도가 너무 낮으면 회귀를 안 하는 로직도 가능
                        # if conf_val < 0.5: continue 
                        
                        reg_model = reg_models[pred_idx]
                        
                        raw_pred = reg_model(curr_x).item()
                        max_p = reg_model.hparams.max_ppm
                        pred_ppm_val = raw_pred * max_p
                        
                        true_ppms.append(true_ppm_val)
                        pred_ppms.append(pred_ppm_val)
                        
                        gas_name = target_gases[true_idx]
                        abs_error = abs(true_ppm_val - pred_ppm_val)
                        
                        results_list.append({
                            "Gas": gas_name,
                            "True_PPM": true_ppm_val,
                            "Pred_PPM": pred_ppm_val,
                            "Confidence": conf_val,
                            "Abs_Error": abs_error
                        })

    # ---------------------------------------------------------
    # 4. 결과 리포트 (신뢰도 통계 추가)
    # ---------------------------------------------------------
    acc = cls_correct / total * 100
    
    # 신뢰도 평균 계산
    avg_conf = np.mean(all_confidences) if all_confidences else 0.0
    
    true_ppms = np.array(true_ppms)
    pred_ppms = np.array(pred_ppms)
    
    if len(true_ppms) > 0:
        rmse = np.sqrt(np.mean((true_ppms - pred_ppms) ** 2))
        # ... (MAPE 계산 로직 유지) ...
        # (생략) 기존 코드와 동일
        epsilon = 1e-7
        mask = true_ppms > 1.0
        if np.sum(mask) > 0:
             masked_true = true_ppms[mask]
             masked_pred = pred_ppms[mask]
             mape = np.mean(np.abs((masked_true - masked_pred) / masked_true)) * 100
             mspe = np.mean(((masked_true - masked_pred) / masked_true) ** 2) * 100
             eval_count = np.sum(mask)
        else:
             mape, mspe = 0.0, 0.0
             eval_count = 0
    else:
        mape, mspe, rmse = 0.0, 0.0, 0.0
        eval_count = 0

    print(f"\n📊 [Final Pipeline Result]")
    print(f"   - Total Samples: {total}")
    print(f"   - Classification Accuracy: {acc:.2f}%")
    print("-" * 40)
    print(f"   🔍 [Confidence Analysis]") # ⚡ 추가된 리포트
    print(f"     • Average Confidence : {avg_conf:.4f}")
    print(f"     • Low Confidence (<{CONF_THRESHOLD}) Count : {low_conf_count} samples")
    print("-" * 40)
    print(f"   - 📉 Regression Metrics:")
    print(f"     • RMSE : {rmse:.4f} ppm")
    print(f"     • MAPE : {mape:.4f} %")
    print(f"     • MSPE : {mspe:.4f} %")
    print("-" * 40)
    
    # ---------------------------------------------------------
    # 5. [수정] Low Confidence 샘플 집중 분석 (교수님 설득용 핵심 자료)
    # ---------------------------------------------------------
    print(f"\n🎨 [Focus Analysis] 신뢰도 {CONF_THRESHOLD} 미만 샘플 분석 중...")
    
    if len(results_list) > 0:
        df_res = pd.DataFrame(results_list)
        
        # ⚡ 1. 임계값 미만 데이터만 필터링 (Low Confidence Only)
        df_low = df_res[df_res["Confidence"] < CONF_THRESHOLD]
        
        low_count = len(df_low)
        total_count = len(df_res)
        ratio = (low_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"   📊 전체 샘플: {total_count}개")
        print(f"   ⚠️ Low Conf 샘플: {low_count}개 ({ratio:.2f}%)")

        if low_count == 0:
            print("   🎉 와우! 설정한 임계값보다 낮은 신뢰도의 샘플이 하나도 없습니다.")
        else:
            # 저장 폴더 분리
            timestamp = datetime.now().strftime('%m%d_%H%M%S')
            save_dir = "./result_pipeline/low_conf_analysis"
            os.makedirs(save_dir, exist_ok=True)
            
            # (1) CSV 저장 (나중에 엑셀 분석용)
            csv_path = os.path.join(save_dir, f"low_conf_samples_{timestamp}.csv")
            df_low.to_csv(csv_path, index=False)
            print(f"   📄 불확실 데이터 저장됨: {csv_path}")

            # -----------------------------------------------
            # 📊 Plot 1: Low Conf 샘플들의 True vs Pred
            # 의도: "신뢰도가 낮은 애들은 실제로도 오차가 크다(대각선에서 멀다)"는 것을 보여줌
            # -----------------------------------------------
            plt.figure(figsize=(8, 8))
            
            # 가스별로 점 모양/색깔 다르게 표시
            sns.scatterplot(
                data=df_low, x="True_PPM", y="Pred_PPM", 
                hue="Gas", style="Gas", s=100, alpha=0.9, palette="Set1"
            )
            
            # 이상적인 정답선 (y=x)
            max_val = max(df_low["True_PPM"].max(), df_low["Pred_PPM"].max())
            plt.plot([0, max_val], [0, max_val], 'k--', label="Ideal Line (Perfect Prediction)")
            
            plt.title(f"Performance on Low Confidence Samples (<{CONF_THRESHOLD})")
            plt.xlabel("True PPM")
            plt.ylabel("Predicted PPM")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(save_dir, f"scatter_low_conf_{timestamp}.png")
            plt.savefig(save_path, dpi=300)
            print(f"   📊 [Plot 1] 저장됨: {save_path}")
            
            # -----------------------------------------------
            # 📊 Plot 2: 어떤 가스를 유독 헷갈려하는가? (Bar Chart)
            # 의도: "특히 벤젠이 헷갈리는 경우가 많으므로 주의해야 합니다" 같은 결론 도출용
            # -----------------------------------------------
            plt.figure(figsize=(8, 5))
            ax = sns.countplot(data=df_low, x="Gas", palette="pastel")
            
            # 막대 위에 숫자 표시
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5),
                            textcoords='offset points')

            plt.title(f"Count of Low Confidence Samples by Gas (<{CONF_THRESHOLD})")
            plt.ylabel("Count")
            
            save_path = os.path.join(save_dir, f"count_low_conf_by_gas_{timestamp}.png")
            plt.savefig(save_path, dpi=300)
            print(f"   📊 [Plot 2] 저장됨: {save_path}")

            # -----------------------------------------------
            # 📊 Plot 3: 가스별 오차 분포 (Box Plot)
            # 의도: "신뢰도가 낮을 때 오차 범위가 얼마나 넓게 퍼지는지(위험도)" 시각화
            # -----------------------------------------------
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df_low, x="Gas", y="Abs_Error", palette="Set2")
            sns.stripplot(data=df_low, x="Gas", y="Abs_Error", color=".3", alpha=0.6) # 점도 같이 찍기
            
            plt.title(f"Absolute Error Distribution (Low Confidence Samples)")
            plt.ylabel("Absolute Error (PPM)")
            
            save_path = os.path.join(save_dir, f"boxplot_error_low_conf_{timestamp}.png")
            plt.savefig(save_path, dpi=300)
            print(f"   📊 [Plot 3] 저장됨: {save_path}")

    else:
        print("⚠️ 추론된 데이터가 없습니다.")
    
        
    
    

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main(args):
    if not os.path.exists(args.save): os.makedirs(args.save)
    
    if args.task == 'reg':
        run_regression(args)
    elif args.task == 'cls':
        run_classification(args)
    elif args.task == 'predict': 
        run_pipeline(args)
    else:
        print("❌ 잘못된 Task입니다. 'reg' 또는 'cls'를 선택하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # [공통 인자]
    parser.add_argument('--task', type=str, required=True, choices=['reg', 'cls','predict'], help='reg: 회귀, cls: 분류')
    parser.add_argument('-mn', '--model_name', type=str, default='cnn1d')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('-dv', '--device', type=str, default='gpu')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')

    # [Regression 전용 인자]
    parser.add_argument('-tg', '--target_gas', type=str, default='benzene', help='회귀 분석 대상 가스')

    # [Classification 전용 인자]
    parser.add_argument('-dt', '--data_type', type=str, default='del', help='pkl 또는 del (분류 데이터 로드 방식)')
    
    parser.add_argument('--ckpt_cls', type=str, help='분류 모델 체크포인트 경로 (.ckpt)')
    
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint', help='회귀 모델들이 있는 폴더 경로')
    
    parser.add_argument('-g', '--gpus', type=int, default=1, help='사용할 GPU 개수 (예: 4)')
    
    parser.add_argument("--model_name_cls", type=str, default="cnncls", help="분류 모델 아키텍처 (예: cnncls)")
    parser.add_argument("--model_name_reg", type=str, default="cnn1d", help="회귀 모델 아키텍처 (예: cnn1d)")
    
    args = parser.parse_args()
    main(args)