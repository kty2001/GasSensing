import subprocess
import time
import sys

# 1. 실행할 모델 목록
models = ["mlp","imlp","cnn1d","cnnnp","icnnnp","cnnd","cnnaa"] 

# 2. 회귀 실험 설정
reg_gases = ["acetone", "benzene", "toluene"]

# 3. 분류 실험 설정 ('del' 데이터 타입 사용 예시)
# cls_data_types = ["del"] 

common_args = ["--epoch", "300", "--batch_size", "128",  "--gpu" ,"4"]

print("🚀 통합 실험 자동화 시작...")

# [Task 1] 회귀 실험 루프
for model in models:
    for gas in reg_gases:
        print(f"\n▶️ [Regression] Model={model}, Gas={gas}")
        cmd = [sys.executable, "main.py", "--task", "reg", "--model_name", model, "--target_gas", gas] + common_args
        subprocess.run(cmd)

# [Task 2] 분류 실험 루프
# for model in models:
#     for dtype in cls_data_types:
#         print(f"\n▶️ [Classification] Model={model}, DataType={dtype}")
#         cmd = [sys.executable, "main.py", "--task", "cls", "--model_name", model, "--data_type", dtype] + common_args
#         subprocess.run(cmd)

print("\n🎉 모든 실험 완료!")