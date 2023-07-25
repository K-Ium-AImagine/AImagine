# 2023년 K-Ium 의료인공지능 경진대회
## > 익명화된 뇌혈관조영술 영상을 기반으로 뇌동맥류 여부, 위치를 진단하는 소프트웨어 개발

![포스터](https://github.com/K-Ium-AImagine/AImagine/assets/90829718/16390247-1bc0-4a3a-a2b6-38a6463cc9d8)

<br />

## 📚 요약
> 이미지 분류 모델 'Efficientnet'을 이용한, 뇌동맥류 진단 AI모델 개발 <br />
> 9,017개의 뇌혈관조영술 영상 이미지 데이터셋과 뇌동맥류 위치 정보를 담고있는 csv파일을 이용하여 학습

<br />

## ⚙ 개발 환경
- Platform : 'Goolge Colab plus'
- 학습 GPU : A100
- 추론 GPU : T4
- Framework : Pytorch
- pretrained model : efficientnet_b0

<br />

## 🔍 1차 검증 결과
> 1차 검증데이터에 대한 AUROC값 <br />
> AUROC of the model : 0.971 <br />
> Accuracy for locations : 0.9638 <br />

![AUC](https://github.com/K-Ium-AImagine/AImagine/assets/90829718/cdd67058-535b-43d6-9c00-6649149a274f)

<br />

## 🛠 학습 & 추론 코드

> ### 학습코드 : [train_model.py](https://github.com/K-Ium-AImagine/AImagine/blob/main/train_model.py)
> ### 추론코드 : [run_inference.py](https://github.com/K-Ium-AImagine/AImagine/blob/main/run_inference.py)
> ### model 디렉토리 : [models](https://github.com/K-Ium-AImagine/AImagine/tree/main/models)

<br />

## 📌 팀 소개
<table>
  <tbody>
    <tr>
      <td align="center"><a href=""><img src="https://avatars.githubusercontent.com/u/90829718?s=400&u=90d56923e2706f34c55a65af5a57da741856d97f&v=4"width="100px;" alt=""/><br /><sub><b> 팀장 : 김동준 </b></sub></a><br /></td>
      <td align="center"><a href=""><img src="https://avatars.githubusercontent.com/u/105621255?v=4" width="100px;" alt=""/><br /><sub><b> 팀원  : 박경민 </b></sub></a><br /></td>
      <td align="center"><a href=""><img src="https://avatars.githubusercontent.com/u/114977536?v=4" width="100px;" alt=""/><br /><sub><b> 팀원 : 박상준 </b></sub></a><br /></td>
      <td align="center"><a href=""><img src="https://avatars.githubusercontent.com/u/113533845?v=4" width="100px;" alt=""/><br /><sub><b> 팀원 : 최유진 </b></sub></a><br /></td>
      </tr>
  </tbody>
</table>
