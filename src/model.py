import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. 모델 팩토리 (통합 진입점)
# -----------------------------------------------------------------------------
def create_model(model: str, input_length: int, output_dim: int):
    """
    args:
        model (str): 모델 이름 ('cnn1d', 'mlp', 'acnn', 'bcnn')
        input_length (int): 입력 데이터 길이 (스펙트럼 포인트 수)
        output_dim (int): 출력 노드 수 (회귀=1, 분류=클래스 개수)
    """
    if model == 'cnn1d':
        return CNN1D(input_length, output_dim)
    elif model == 'mlp':
        return MLP(input_length, output_dim)
    elif model == 'imlp':
        return IMLP(input_length, output_dim)
    elif model == 'cnnnp':
        return CNNNP(input_length, output_dim)
    elif model == 'icnnnp':
        return ICNNNP(input_length, output_dim)
    elif model == 'cnnd':
        return CNNDilation(input_length, output_dim)
    elif model == 'cnnaa':
        return CNN1Daa(input_length, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model}")


# -----------------------------------------------------------------------------
# 2. [Upgrade] CNN1D (파라미터 대폭 증가 버전)
#    - 기존 300K -> 수백만(M) 단위로 체급을 키워 MLP와 대등하게 만듦
# -----------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            
            # Block 2
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            
            # Block 3
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, output_dim) # 1 or 3
        )

    def forward(self, x):
        # 입력 차원 보정: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# 2. MLP 
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        if x.dim() == 3: # (Batch, 1, Length) -> (Batch, Length)
            x = x.squeeze(1)
        return self.net(x)

class IMLP(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        if x.dim() == 3: # (Batch, 1, Length) -> (Batch, Length)
            x = x.squeeze(1)
        return self.net(x)

    
class CNNNP(nn.Module):
    def __init__(self, input_length: int, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * input_length, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.fc(x)


class ICNNNP(nn.Module):
    def __init__(self, input_length: int, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * input_length, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.fc(x)
    
class CNNDilation(nn.Module):
    def __init__(self, input_length: int, output_dim: int):
        super().__init__()
        

        self.features = nn.Sequential(
            # Conv1 
            nn.Conv1d(1, 16, kernel_size=7, padding=3, dilation=1),
            nn.ReLU(),   
            
            # Conv2 
            nn.Conv1d(16, 32, kernel_size=7, padding=6, dilation=2),
            nn.ReLU(),   
            
            # Conv3 
            nn.Conv1d(32, 64, kernel_size=7, padding=12, dilation=4),
            nn.ReLU(),   
            
            # Flatten & FC
            nn.Flatten(), 
            nn.Linear(64 * input_length, output_dim) 
        )

    def forward(self, x):
        # 입력 차원 보정 (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        return self.features(x)
    

class CNN1Daa(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=127, padding=63),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 4
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 5 (깊이 추가)
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) 
        )

        # [중요] 모델 정의가 끝난 후 초기화 함수 호출
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming He Initialization을 모델의 모든 레이어에 적용합니다.
        """
        for m in self.modules():
            # 1. Convolution Layer 초기화
            if isinstance(m, nn.Conv1d):
                # 가중치: He Normal (ReLU에 최적화)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 편향(Bias): 존재한다면 0으로 초기화
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
           
    def forward(self, x):
        # 입력 차원 보정: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
