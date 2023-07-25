
# < 학습 & 추론코드 작성환경 >
"""
Platform : 'Goolge Colab'
학습 GPU: A100
추론 GPU: T4
GPU API: cuda
"""

# Language 
"""
Python
    version: 3.10.12 (main, Jun  7 2023, 12:45:35) [GCC 9.4.0]
"""

# Library
"""
1. Pytorch
version: 2.0.1+cu118
License: BSD

2. Pandas
version: 1.5.3
License: BSD 3-Clause

3. Numpy
version: 1.22.4
License: BSD (NumPy license)

4. PIL
version: 8.4.0
License: HPND

5. Scikit-learn
version: 1.2.2
License: BSD (new BSD)

6. Matplotlib
version: 3.7.1
License: Matplotlib License (PSF-based)

7. Torchvision
version: 0.15.2+cu118
License: BSD 3-Clause

CUDA version: 11.8
CUDNN version: 8700

 < pretrained model >
    model: efficientnet_b0
    License: Apache 2.0

"""


# Library install 가이드라인
"""
아래의 순서로 설치 해주세요.
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas==1.5.3
pip install numpy==1.22.4
pip install pillow==8.4.0
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.1
pip install efficientnet_pytorch
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
from PIL import Image
import sys
import matplotlib
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
# 라이브러리 버전 확인 (필요시 주석 제거후 실행)
"""
print(f"Python version: {sys.version}") # Python version: 3.10.12 (main, Jun  7 2023, 12:45:35) [GCC 9.4.0]
print(f"PyTorch version: {torch.__version__}") # PyTorch version: 2.0.1+cu118
print(f"Pandas version: {pd.__version__}") # Pandas version: 1.5.3
print(f"Numpy version: {np.__version__}") # Numpy version: 1.22.4
print(f"PIL version: {Image.__version__}") # PIL version: 8.4.0
print(f"Scikit-learn version: {sklearn.__version__}")  # Scikit-learn version: 1.2.2
print(f"Matplotlib version: {matplotlib.__version__}")  # Matplotlib version: 3.7.1
print(f"Torchvision version: {torchvision.__version__}") # Torchvision version: 0.15.2+cu118
print(f"CUDA version: {torch.version.cuda}") # CUDA version: 11.8
print(f"CUDNN version: {torch.backends.cudnn.version()}") # CUDNN version: 8700
"""

### (필독) 경로변수 설정 가이드라인 ###
"""
> 입력 경로
    1. train_csv_path : 모델 학습에 사용할, train.csv파일의 경로를 입력해주세요.
        예시 ) '/content/drive/MyDrive/Colab Notebooks/AImagine/2023_k_ium_composition/train_set/train.csv'
        ※ .csv파일을 최종 타겟으로 설정해주세요.
    2. img_dir_path : 모델이 학습할, 이미지 데이터셋이 위치한 디렉토리 경로를 입력해주세요.
        예시 ) '/content/drive/MyDrive/Colab Notebooks/AImagine/2023_k_ium_composition/train_set/'
        ※ 디렉토리 경로를 최종 타겟으로 설정해주세요.
> 출력 경로
    학습완료 후 생성되는 모델들은 현재 작업경로 하위 models 디렉토리를 생성하여, 위치열 단위로 저장됩니다(./models/{target_column}/)
    여기서 target_column은 학습시 각 타겟 위치열을 뜻합니다. 이는 각 부분마다 설정되어 있습니다.

그 외에 학습을 원하는 위치열 target_column을 입력하면 학습 진행이 가능합니다.
※ 'BA'위치열에 대한 학습부분은 하단에 별도로 존재합니다.

"""

## 경로변수 설정 (위의 가이드를 참고하여 설정해주세요.) ##

# 입력 경로
train_csv_path = ''
img_dir_path = ''


# Train / Validation 데이터셋 나누기
data = pd.read_csv(train_csv_path)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

val_data.to_csv('ValidationSet.csv', index=False)
train_data.to_csv('TrainSet.csv', index=False)

# GPU 사용설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 640 * 640 이미지 사이즈 통일
    transforms.ToTensor(),  # 이미지를 PyTorch tensor로 변환
])

# 데이터셋 클래스 정의
class BrainDataset(Dataset):
    def __init__(self, data, root_dir, target_column, position, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.target_column = target_column  # 학습타겟 위치열 
        self.position = position
        # 원-핫 인코딩 기법
        self.pos_encoding = {
            "LI-A": [1, 0, 0, 0, 0, 0, 0, 0],
            "LI-B": [0, 1, 0, 0, 0, 0, 0, 0],
            "LV-A": [0, 0, 1, 0, 0, 0, 0, 0],
            "LV-B": [0, 0, 0, 1, 0, 0, 0, 0],
            "RI-A": [0, 0, 0, 0, 1, 0, 0, 0],
            "RI-B": [0, 0, 0, 0, 0, 1, 0, 0],
            "RV-A": [0, 0, 0, 0, 0, 0, 1, 0],
            "RV-B": [0, 0, 0, 0, 0, 0, 0, 1]
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지 index 설정
        img_index = self.data.iloc[idx, 0]

        # 매개변수로 받아온 위치 position에 따른 이미지 파일명 설정
        pos = self.position
        img_name = os.path.join(self.root_dir, str(img_index) + pos + ".jpg")

        # 이미지를 불러와서 tensor로 변환
        image = Image.open(img_name)
        image = self.transform(image)

        # 원-핫 인코딩 벡터를 설정
        pos_vector = self.pos_encoding[pos]

        # 라벨 설정
        label = torch.as_tensor(self.data.iloc[idx][self.target_column])  # 선택한 레이블만 반환

        return image, torch.tensor(pos_vector, dtype=torch.float32), label  # 원-핫 인코딩 벡터 추가


# 모델 클래스 정의
class SingleImageEfficientNet(nn.Module):
    def __init__(self, output_dim):
        super(SingleImageEfficientNet, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()  # 마지막 classifier 레이어 제거 (특징만 추출)
        self.dropout = nn.Dropout(0.2)  # 드롭아웃 비율 0.2 설정
        self.batch_norm = nn.BatchNorm1d(1280 + 8)  # Batch Normalization layer 추가, 원-핫 인코딩 벡터 차원만큼 증가
        self.fc = nn.Linear(1280 + 8, output_dim)  # 모델의 출력을 FC layer 통과, 원-핫 인코딩 벡터 차원만큼 증가

    def forward(self, x, pos_vector):
        x = self.model(x)  # 모델에 이미지를 전달하여 특징 추출
        x = x.view(x.size(0), -1)
        x = torch.cat((x, pos_vector), dim=1)  # 특징과 원-핫 인코딩 벡터를 연결
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.batch_norm(x)  # Batch Normalization 적용
        x = self.fc(x)  # 특징을 FC layer 통과
        return x


# 학습할 target_column 설정

# 'Aneurysm', 'BA' 제외
"""
    아래의 20가지 위치열들을 target_column으로 설정하여, 학습을 진행하세요.
    1. L_ICA    2. R_ICA    3. L_PCOM   4. R_PCOM   5.L_AntChor   6. R_AntChor   7. L_ACA    8. R_ACA    9. L_ACOM   10. R_ACOM
    11. L_MCA   12. R_MCA   13. L_VA    14. R_VA    15. L_PICA    16. R_PICA     17. L_SCA   18. R_SCA   19. L_PCA    20. R_PCA
"""
target_column = ' ' # 타겟 위치열 입력하기

# 클래스 가중치 조정
class_counts = train_data[target_column].value_counts().to_dict()
print(class_counts)

total_counts = sum(class_counts.values())
class_weights = {cls: total_counts / count for cls, count in class_counts.items()}

weights = [class_weights[cls] for cls in sorted(class_weights.keys())]
weights = torch.tensor(weights, dtype=torch.float).to(device)


# 학습 코드

""" 
target_column에 따라 구별하여, 이미지 데이터셋의 왼쪽(LI-A, LI-B, LV-A, LV-B) 또는 오른쪽(RI-A, RI-B, RV-A, RV-B)을 사용할지 결정하여 학습합니다.
학습 후, 생성된 모델들 중 best모델을 선정하여 저장합니다. ('BA' 위치열 제외)
"""
n_epochs_stop = 5  # Early stopping 기준점
learning_rate = 0.001  # 학습률 조정기법을 고려한 0.001로 시작
batch_size = 32  # A100 GPU기준 64부터, GPU 메모리 이슈 발생

# Loss 기록을 위한 리스트 생성
train_losses = []
val_losses = []

# target_column별로 모델 저장할 디렉토리 경로 설정
model_dir = f'./models/{target_column}/'
os.makedirs(model_dir, exist_ok=True)  # 해당 디렉토리가 없으면 생성

epochs_no_improve = 0
early_stop = False
min_val_loss = np.Inf

# 데이터셋 생성
img_dir = img_dir_path # 학습에 사용할, 이미지 데이터셋이 위치한 "디렉토리 경로"를 입력 / 상단의 경로변수 설정 필수(img_dir_path)

best_models = {}  # 각 위치에 대한 최고의 모델을 저장하는 딕셔너리

# target_column에 따른, 왼쪽(L)과 오른쪽(R)중 선택
vessels = ['LI', 'LV'] if target_column[0] == 'L' else ['RI', 'RV']

for vessel in vessels:
    for angle in ['A', 'B']:
        position = vessel + '-' + angle

        if target_column[0] == 'L' and vessel not in ['LI', 'LV']:
            print("왼쪽 아니기에 skip")
            continue

        if target_column[0] == 'R' and vessel not in ['RI', 'RV']:
            print("오른쪽 아니기에 skip")
            continue

        print(position)

        # 학습 및 검증 데이터셋 생성
        train_dataset = BrainDataset(train_data, root_dir=img_dir, target_column=target_column, position=position, transform=transform)
        val_dataset = BrainDataset(val_data, root_dir=img_dir, target_column=target_column, position=position, transform=transform)

        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

        # 모델 및 최적화 도구 초기화
        model = SingleImageEfficientNet(output_dim=1).to(device)
        pos_weight = torch.tensor([class_weights[1]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        best_val_loss = float('inf')

        epochs_no_improve = 0
        early_stop = False

        # 학습 시작
        num_epochs = 25
        for epoch in range(num_epochs):
            # Train
            model.train()
            for inputs, pos_vector, labels in train_loader:
                inputs = inputs.to(device)
                pos_vector = pos_vector.to(device)
                labels = labels.to(device).unsqueeze(1).float()

                outputs = model(inputs, pos_vector)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())  # Train loss 기록

            # Validation
            model.eval()
            val_preds = []
            val_labels_list = []
            with torch.no_grad():
                for val_inputs, val_pos_vector, val_labels in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_pos_vector = val_pos_vector.to(device)
                    val_labels = val_labels.to(device).unsqueeze(1).float()

                    val_outputs = model(val_inputs, val_pos_vector)
                    val_loss = criterion(val_outputs, val_labels)
                    val_preds.append(val_outputs.sigmoid().cpu().numpy())
                    val_labels_list.append(val_labels.cpu().numpy())

                    val_losses.append(val_loss.item())  # Validation loss 기록

                val_preds = np.concatenate(val_preds)
                val_labels = np.concatenate(val_labels_list)
                val_auc = roc_auc_score(val_labels, val_preds)
                print(f"{position} : Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation AUC: {val_auc}")

                # 학습률 조정
                scheduler.step(val_loss.item())

                # Epoch별 모델 저장
                torch.save(model.state_dict(), os.path.join(model_dir, f'{position}_model_{epoch+1}.pth'))

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_models[position] = model.state_dict()
                    # Best 모델 저장
                    torch.save(model.state_dict(), os.path.join(model_dir, f'{position}_best_model.pth'))
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == n_epochs_stop:
                        print('Early stopping!')
                        early_stop = True
                        break
        if early_stop:
            print("Stopped")


# Best모델간 최종 모델 선정
best_model_overall = None
best_auc = -1
for position, model_state_dict in best_models.items():
    model = SingleImageEfficientNet(output_dim=1).to(device)
    model.load_state_dict(model_state_dict)

    val_dataset = BrainDataset(val_data, root_dir=img_dir, target_column=target_column, position=position, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    model.eval()
    val_preds = []
    val_labels_list = []
    with torch.no_grad():
        for val_inputs, val_pos_vector, val_labels in val_loader:
            val_inputs = val_inputs.to(device)
            val_pos_vector = val_pos_vector.to(device)
            val_labels = val_labels.to(device).unsqueeze(1).float()

            val_outputs = model(val_inputs, val_pos_vector)
            val_preds.append(val_outputs.sigmoid().cpu().numpy())
            val_labels_list.append(val_labels.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels_list)
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"{target_column} : {position} model Validation AUC: {val_auc}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_overall = model
            # 최종 모델 저장
            torch.save(model.state_dict(), os.path.join(model_dir, f'{target_column}_best_model.pth')) 


## 'BA' 위치열 학습코드 ##
"""
BA(기저동맥)은 좌우 좌/우 구별없이, 추골동맥 이미지 데이터셋를 이용하여 학습합니다.
"""

# 'BA'용 데이터셋 재정의
class BrainDataset(Dataset):
    def __init__(self, data, root_dir, target_column, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.target_column = target_column
        self.pos_encoding = {
            "LI-A": [1, 0, 0, 0, 0, 0, 0, 0],
            "LI-B": [0, 1, 0, 0, 0, 0, 0, 0],
            "LV-A": [0, 0, 1, 0, 0, 0, 0, 0],
            "LV-B": [0, 0, 0, 1, 0, 0, 0, 0],
            "RI-A": [0, 0, 0, 0, 1, 0, 0, 0],
            "RI-B": [0, 0, 0, 0, 0, 1, 0, 0],
            "RV-A": [0, 0, 0, 0, 0, 0, 1, 0],
            "RV-B": [0, 0, 0, 0, 0, 0, 0, 1]
        }

    def __len__(self):
        return len(self.data) * 4   # 4개의 이미지 사용

    def __getitem__(self, idx):
        patient_index = idx // 4
        img_index = self.data.iloc[patient_index, 0]

        pos = ["LV-A", "LV-B", "RV-A", "RV-B"][idx % 4]     # 추골동맥 이미지 사용
        img_name = os.path.join(self.root_dir, str(img_index) + pos + ".jpg")

        # 이미지를 불러와서 tensor로 변환
        image = Image.open(img_name)
        image = self.transform(image)

        # 원-핫 인코딩 벡터를 얻음
        pos_vector = self.pos_encoding[pos]

        # 라벨을 얻음
        label = torch.as_tensor(self.data.iloc[patient_index][self.target_column])  # 선택한 레이블만 반환

        return image, torch.tensor(pos_vector, dtype=torch.float32), label  # 원-핫 인코딩 벡터 추가


# 학습 시작
target_column = 'BA'    # 고정

n_epochs_stop = 5
learning_rate = 0.001
batch_size = 32 

# Loss 기록을 위한 리스트 생성
train_losses = []
val_losses = []

model_dir = f'./models/{target_column}/'
os.makedirs(model_dir, exist_ok=True)

epochs_no_improve = 0
early_stop = False
min_val_loss = np.Inf

# 데이터셋 생성
img_dir = img_dir_path # 학습에 사용할, 이미지 데이터셋이 위치한 "디렉토리 경로"를 입력 / 상단의 경로변수 설정 필수(img_dir_path)
train_dataset = BrainDataset(train_data, root_dir=img_dir, target_column=target_column, transform=transform)
val_dataset = BrainDataset(val_data, root_dir=img_dir, target_column=target_column, transform=transform)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

model = SingleImageEfficientNet(output_dim=1).to(device)
model = model.to(device)


pos_weight = torch.tensor([class_weights[1]]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
best_val_loss = float('inf')
# 스케줄러 생성
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# 학습 시작
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    for inputs, pos_vector, labels in train_loader:
        inputs = inputs.to(device)
        pos_vector = pos_vector.to(device)
        labels = labels.to(device).unsqueeze(1).float()

        outputs = model(inputs, pos_vector)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())  # Train loss 기록

    model.eval()
    val_preds = []
    val_labels_list = []
    with torch.no_grad():
        for val_inputs, val_pos_vector, val_labels in val_loader:
            val_inputs = val_inputs.to(device)
            val_pos_vector = val_pos_vector.to(device)
            val_labels = val_labels.to(device).unsqueeze(1).float()

            val_outputs = model(val_inputs, val_pos_vector)
            val_loss = criterion(val_outputs, val_labels)
            val_preds.append(val_outputs.sigmoid().cpu().numpy())
            val_labels_list.append(val_labels.cpu().numpy())

            val_losses.append(val_loss.item())  # Validation loss 기록

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels_list)
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"{target_column} :Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation AUC: {val_auc}")

        # 학습률 조정
        scheduler.step(val_loss.item())

        torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch+1}.pth'))

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            # Best 모델 저장
            torch.save(model.state_dict(), os.path.join(model_dir, f'{target_column}_best_model.pth'))            
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                early_stop = True
                break
if early_stop:
    print("Stopped")

class LowOutputModel(nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), 1)

Low_model = LowOutputModel().to(device)
model_dir = f'./models/{target_column}/'
os.makedirs(model_dir, exist_ok=True)

# 'L_PICA', 'L_SCA', 'L_PCA', 'R_PCA'에 대한 Low 모델 pth 파일 생성
Low_model_columns = ['L_PICA', 'L_SCA', 'L_PCA', 'R_PCA']
for target_column in Low_model_columns:
    torch.save(Low_model.state_dict(), os.path.join(model_dir, f'Low_model_{target_column}.pth'))
