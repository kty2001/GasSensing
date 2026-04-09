import torch
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from src.model import create_model
from src.utils import build_samples, SEED

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Device set to: {device}")
    
    # ---------------------------------------------------------
    # 1. 실제 데이터 로드 (main.py와 동일한 방식)
    # ---------------------------------------------------------
    print("\n📂 실제 데이터(Test Set) 로드 중...")
    X, y_index, _ = build_samples('del') # 데이터 타입에 맞게 변경 (예: 'del')
    
    _, X_test, _, _ = train_test_split(
        X, y_index, test_size=0.2, random_state=SEED, stratify=y_index
    )
    
    # PyTorch Tensor로 변환
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    num_samples = len(X_test_tensor)
    print(f"✅ 테스트 셋 로드 완료: 총 {num_samples}개 샘플 (Real Data)")

    # ---------------------------------------------------------
    # 2. 벤치마크 설정
    # ---------------------------------------------------------
    models_list = ["mlp", "imlp", "cnn1d", "cnnnp", "icnnnp", "cnnd", "cnnaa"]
    results = []
    
    print("\n🚀 아키텍처별 실제 데이터 스트리밍 추론 속도 측정 시작...\n")
    
    for model_name in models_list:
        print(f"⏳ Testing [{model_name.upper()}] ...", end=" ", flush=True)
        
        # 모델 껍데기 로드 (속도는 구조에 의해서만 결정되므로 가중치 로드 생략)
        cls_model = create_model(model_name, 7300, 3).to(device).eval()
        
        # 회귀 모델 3개 (아세톤, 벤젠, 톨루엔) 로드하여 파이프라인 완벽 모사
        reg_models = {
            0: create_model(model_name, 7300, 1).to(device).eval(),
            1: create_model(model_name, 7300, 1).to(device).eval(),
            2: create_model(model_name, 7300, 1).to(device).eval()
        }
        
        # Warm-up (초기 지연시간 제거)
        dummy_x = X_test_tensor[0].unsqueeze(0)
        with torch.no_grad():
            for _ in range(50):
                pred_cls = torch.argmax(cls_model(dummy_x), dim=1).item()
                _ = reg_models[pred_cls](dummy_x)
        
        # 실제 데이터 추론 측정 (센서에서 실시간으로 1개씩 들어오는 상황 모사)
        pipeline_times = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # 데이터 1개 추출 (Batch Size = 1)
                curr_x = X_test_tensor[i].unsqueeze(0)
                
                if device == 'cuda':
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()
                    
                    # [파이프라인 실행] 분류 -> 판단 -> 해당 회귀 모델 실행
                    logits = cls_model(curr_x)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    _ = reg_models[pred_idx](curr_x)
                    
                    ender.record()
                    torch.cuda.synchronize()
                    pipeline_times.append(starter.elapsed_time(ender))
                else:
                    start_t = time.perf_counter()
                    
                    logits = cls_model(curr_x)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    _ = reg_models[pred_idx](curr_x)
                    
                    pipeline_times.append((time.perf_counter() - start_t) * 1000)
                    
        # 통계 계산
        avg_time = np.mean(pipeline_times)
        std_time = np.std(pipeline_times)
        fps = 1000 / avg_time
        
        print("완료!")
        
        results.append({
            "Model": model_name.upper(),
            "Total Samples Evaluated": num_samples,
            "Avg Pipeline Time (ms)": round(avg_time, 3),
            "Std Dev (ms)": round(std_time, 3),
            "Max FPS (초당 처리량)": round(fps, 0)
        })
        
    # ---------------------------------------------------------
    # 3. 결과 출력 및 저장
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 [Real Data Pipeline Speed Benchmark Results]")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    csv_path = "speed_benchmark_real_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 실제 데이터 기반 벤치마크 결과가 '{csv_path}'로 저장되었습니다.")

if __name__ == "__main__":
    main()
    