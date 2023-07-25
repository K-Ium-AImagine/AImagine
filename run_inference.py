
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

5. Torchvision
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
pip install efficientnet_pytorch
"""

import os
import glob
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import pandas as pd
import numpy as np

import sys
# 라이브러리 버전 확인 (필요시 주석 제거후 실행)
"""
print(f"Python version: {sys.version}") # Python version: 3.10.12 (main, Jun  7 2023, 12:45:35) [GCC 9.4.0]
print(f"PyTorch version: {torch.__version__}") # PyTorch version: 2.0.1+cu118
print(f"Pandas version: {pd.__version__}") # Pandas version: 1.5.3
print(f"Numpy version: {np.__version__}") # Numpy version: 1.22.4
print(f"PIL version: {Image.__version__}") # PIL version: 8.4.0
print(f"Torchvision version: {torchvision.__version__}") # Torchvision version: 0.15.2+cu118
print(f"CUDA version: {torch.version.cuda}") # CUDA version: 11.8
print(f"CUDNN version: {torch.backends.cudnn.version()}") # CUDNN version: 8700
"""

### (필독) 경로변수 설정 가이드라인 ###
"""
> 입력 경로

  1. models_path : 제출물에서 .pth 형식 21개의 모델이 위치한 'models'디렉토리 경로를 설정해주세요.
        예시 ) '/content/drive/MyDrive/Colab Notebooks/AImagine/models/*.pth' 
        ※ 경로 끝은 /*.pth 형식을 지켜주세요.

  2. test_csv_path : 모델의 추론값이 입력될, 0으로 코딩되어있는 test.csv파일의 경로를 설정해주세요.
        예시 ) '/content/drive/MyDrive/Colab Notebooks/AImagine/2023_k_ium_composition/test_set/test.csv'
        ※ .csv파일을 최종 타겟으로 설정해주세요.

  3. img_dir_path : 모델이 적용될, 이미지 데이터셋이 위치한 디렉토리 경로를 입력해주세요.
        예시 ) '/content/drive/MyDrive/Colab Notebooks/AImagine/2023_k_ium_composition/train_set/'
        ※ 디렉토리 경로를 최종 타겟으로 설정해주세요.

> 출력 경로

  1. output_dir_path : 추론 완료 후, output.csv파일을 저장할 디렉토리 경로를 입력해주세요.
        예시 ) '/content/drive/MyDrive/Colab Notebooks/AImagine/2023_k_ium_composition/'
        ※ 디렉토리 경로를 최종 타겟으로 설정해주세요. 경로 미입력시, 현재 작업경로에 저장됩니다.
"""

## 경로변수 설정 (위의 가이드를 참고하여 설정해주세요.) ##

# 입력 경로
models_path = ''    # 167번 라인에서 적용
test_csv_path = ''  # 191번 라인에서 적용
img_dir_path = ''  # 193번 라인에서 적용

# 출력 경로
output_dir_path = '' # 258번 라인에서 적용
if not output_dir_path:  # output.csv 출력경로 미설정시, 현재 작업경로에 저장
    output_dir_path = os.getcwd()


# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),  # 이미지를 PyTorch tensor로 변환
])

# 위치 인코딩 정보를 저장하는 딕셔너리
pos_encoding = {
    "LI-A": [1, 0, 0, 0, 0, 0, 0, 0],
    "LI-B": [0, 1, 0, 0, 0, 0, 0, 0],
    "LV-A": [0, 0, 1, 0, 0, 0, 0, 0],
    "LV-B": [0, 0, 0, 1, 0, 0, 0, 0],
    "RI-A": [0, 0, 0, 0, 1, 0, 0, 0],
    "RI-B": [0, 0, 0, 0, 0, 1, 0, 0],
    "RV-A": [0, 0, 0, 0, 0, 0, 1, 0],
    "RV-B": [0, 0, 0, 0, 0, 0, 0, 1]
}

# 모델 구조 정의
class SingleImageEfficientNet(nn.Module):
    def __init__(self, output_dim):
        super(SingleImageEfficientNet, self).__init__()
        # pre_trained 모델 : efficientnet_b0    가중치 : EfficientNet_B0_Weights.IMAGENET1K_V1 
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()  # 마지막 classifier 레이어 제거 (특징만 추출)
        self.dropout = nn.Dropout(0.2)  # 드롭아웃 비율을 0.2로 조정
        self.batch_norm = nn.BatchNorm1d(1280 + 8)  # Batch Normalization layer 추가, 원-핫 인코딩 벡터 차원만큼 증가
        self.fc = nn.Linear(1280 + 8, output_dim)  # 모델의 출력을 FC layer 통과, 원-핫 인코딩 벡터 차원만큼 증가

    def forward(self, x, pos_vector):
        x = self.model(x)  # 모델에 이미지를 전달하여 특징을 추출
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat((x, pos_vector), dim=1)  # 특징과 원-핫 인코딩 벡터를 연결
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.batch_norm(x)  # Batch Normalization 적용
        x = self.fc(x)  # 특징을 FC layer 통과
        return x

class LowOutputModel(nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), 1)
    

# 모델 불러오기

low_models = {'L_PICA', 'L_SCA', 'L_PCA', 'R_PCA'}
models = {}
for model_path in glob.glob(models_path):  # 추론에 사용할 model들 load / 상단의 경로변수 설정 필수(models_path)
    print(f"Processing {model_path}")  # 현재 load 중인 모델 파일
    model_name = os.path.basename(model_path).split('.')[0]
    if '+' in model_name:
        target_column, img_name = model_name.split('+')
    else:  # BA 모델 처리
        target_column = model_name
        img_name = None

    if target_column in low_models:
        model = LowOutputModel()
    else:
        model = SingleImageEfficientNet(output_dim=1).to(device)
        model.load_state_dict(torch.load(model_path))

    model.eval()
    models[target_column] = {'model': model, 'img_name': img_name}


print(models.keys())

## 테스트 데이터 로드 및 추론 ##

# 0으로 코딩되어있는 기본 test.csv 파일 load / # 상단의 경로변수 설정 필수(test_csv_path) 
test_data = pd.read_csv(test_csv_path)
# 추론할 이미지 데이터셋이 위치한 디렉토리 경로 설정 / # 상단의 경로변수 설정 필수(img_dir_path)
img_dir = img_dir_path

for idx, row in test_data.iterrows():
    img_index = str(row['Index']).zfill(4)
    for column in test_data.columns[1:]:
        if column == 'Aneurysm':
            continue
        elif column == 'BA':  # BA 위치열에 대한 처리
            preds = []
            for img_name in ['LV-A', 'LV-B', 'RV-A', 'RV-B']:
                img_path = os.path.join(img_dir, img_index + img_name + '.jpg')
                img = Image.open(img_path)
                img = transform(img).unsqueeze(0).to(device)

                pos_vector = torch.tensor(pos_encoding[img_name], dtype=torch.float32).unsqueeze(0).to(device)
                output = models[column]['model'](img, pos_vector)  # pos_vector 추가
                output_prob = torch.sigmoid(output).item()
                output_prob = round(output_prob, 3)
                preds.append(output_prob)
            test_data.loc[idx, column] = max(preds) if max(preds) > 0.5 else np.mean(preds)
        else:
            img_name = models[column]['img_name']
            if img_name:
                img_path = os.path.join(img_dir, img_index + img_name + '.jpg')
                img = Image.open(img_path)
                img = transform(img).unsqueeze(0).to(device)

                pos_vector = torch.tensor(pos_encoding[img_name], dtype=torch.float32).unsqueeze(0).to(device)
                if isinstance(models[column]['model'], LowOutputModel):
                  output = models[column]['model'](img)
                  output_prob = output.item()
                  output_prob = round(output_prob, 3)
                else:
                  output = models[column]['model'](img, pos_vector)
                  output_prob = torch.sigmoid(output).item()
                  output_prob = round(output_prob, 3)

                test_data.loc[idx, column] = output_prob



# 추론 시작

for idx, row in test_data.iterrows():
    # low_models 제외한 위치열들만 선택하여, 그 값들 중에서 0.67을 넘는 값들을 추출.
    over_values = row[[col for col in test_data.columns[1:] if col not in low_models]].values
    over_values = over_values[over_values > 0.67]

    # over_values를 넘는 값이 존재한다면 그 값들 중 가장 큰 값을 Aneurysm에 입력하고, over_values를 넘는 값이 없다면 값들 중에 가장 낮은 값을 Aneurysm에 입력.
    if over_values.size > 0:
        test_data.loc[idx, 'Aneurysm'] = np.max(over_values)
    else:
        non_low_model_values = row[[col for col in test_data.columns[2:] if col not in low_models]]
        test_data.loc[idx, 'Aneurysm'] = np.min(non_low_model_values)

test_data['Aneurysm'] = test_data['Aneurysm'].apply(lambda x: 0.999 if x == 1 else (0.001 if x == 0 else x))

for column in test_data.columns[1:]:  # 'Aneurysm' 열 제외
    if column == 'Aneurysm':
        continue
    else:
        test_data[column] = test_data[column].apply(lambda x: 1 if x >= 0.67 else 0)

# 결과를 CSV 파일로 저장 / # 상단의 경로변수 설정 필수(output_dir_path)
output_csv_path = os.path.join(output_dir_path, 'output.csv')
test_data.to_csv(output_csv_path, index=False)

print("추론 완료.")