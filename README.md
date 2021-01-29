# Recommender system for advertisement
This repository for my master's thesis "A Deep Learning-Based System for Recommending Advertisement Through Real-Time User Face Recogniiton" in Kyung Hee University at Seoul, Korea.

# 논문 개요
<p align="center"><img src="https://user-images.githubusercontent.com/35788594/105966523-1fe8ac80-60c8-11eb-9c94-c43edd7f453f.png" width="60%"></p>

광고를 시청하는 사용자의 얼굴을 실시간 인식하여 1) 광고에 대한 평점 예측 2) 새로운 광고 추천을 함

## 데이터 수집 및 전처리

### 데이터 수집기
<p align="center"><img src="https://user-images.githubusercontent.com/35788594/105967857-af428f80-60c9-11eb-8a8c-bbfc57113dd0.png"></p>

데이터 수집기는 pyQt를 이용하여 UI로 제작하여 실험을 비대면으로 진행하였으며, COVID-19로 인한 데이터 수집 문제를 해결하였다.
수집기를 실행하고 각 video 버튼을 누르면 해당되는 광고 영상이 재생되며, 재생되는 동안 사용자의 얼굴이 웹캠을 통해 촬영 및 저장된다.

### 데이터 수집 결과
<p align="center"><img src="https://user-images.githubusercontent.com/35788594/105968372-5cb5a300-60ca-11eb-9cf2-53b58dbb1794.png" width="40%"></p>

데이터는 광고 영상 20개에 대한 사용자의 얼굴이 담긴 영상이며, 총 77명 (남 39명, 여 38명)에 대해 수집하였다.

## 실험 방법론

### 평점 예측
사용자의 얼굴 변화량을 얼굴 이미지 행렬 값의 변화량으로 정의하고, 변화량 이미지를 RestNet-18 신경망으로 훈련시켜 평점을 예측하였다.
예측 평점은 (이전 시간동안 신경망이 예측한 평점의 평균)으로 계산하였다.

#### 성능 평가 지표
Mean Absolute Error (MAE)를 사용하였다.
<p align="center"><img src="https://user-images.githubusercontent.com/35788594/106219306-723ae200-621c-11eb-95e2-a7e6a6e9bfd4.png" width="40%"></p>

#### 비교 모델
1) UserCF-based rating prediction system
2) Average rating prediction system
3) Random rating prediction system

### 광고 추천
Keypoint matching을 통해 사용자의 얼굴 변화도를 평가하였으며, 변화도가 유사한 사용자를 search하여, search한 유사한 사용자가 선호하는 광고를 추천하였다.

### 성능 평가 지표
추천한 광고 N개와 실제 사용자가 선호하는 광고 N개 간 겹치는 개수에 대한 비율을 Recommend Hit Ratio (RHR)로 정의하여, 성능을 비교 평가하였다. 
<p align="center"><img src="https://user-images.githubusercontent.com/35788594/106219362-8e3e8380-621c-11eb-90d8-053ff31a55dd.png" width="40%"></p>


#### 비교 모델
1) UserCF-based recommender system
2) Best-selling recommender system
3) Random recommender system

## 실험 결과

### 평점 예측
<p align="center"><img src="https://user-images.githubusercontent.com/35788594/106218915-b11c6800-621b-11eb-8552-2de28acb6266.png" width="60%"></p>

성능이 더 우수하거나, 기존과 큰 차이를 보이지 않았다.
하지만, 본 논문의 제안 방법론은 새로운 사용자의 얼굴을 추천 매개체로 사용하기 때문에, 기존 사용자 및 광고에 대한 선호도 정보를 가지고 있는 모델들과 성능이 유사하다는 것은 큰 의의를 가진다.

### 광고 추천
<p align="center"><img src="https://user-images.githubusercontent.com/35788594/106219193-3ef85300-621c-11eb-864a-415c127be1ef.png" width="60%"></p>

광고 1개를 추천했을 때 성능이 기존 모델보다 성능이 좋았으나, 추천하는 광고의 개수가 많아질수록 성능이 유사해짐을 알 수 있다.
하지만, 본 논문의 제안 방법론은 광고를 시청하는 사용자에게 가장 적합한 광고 하나를 추천하여, 추천 광고를 새로 재생시키려는 목적을 가지고 있기 때문에, 연구 목적에 맞게 결과가 나왔음을 알 수 있다.

## 결론
CNN을 이용한 평점 예측 및 Keypoint matching을 이용한 광고 추천 시스템을 개발하였다.
앞서 나온 두가지 결과를 통해, 본 연구 목적에 맞는 결과를 도출하였다.
또한, 기존 추천 시스템의 문제점인 Cold-start 문제와 Long-tail 문제를 해결하였다.

### 추후 연구 목표
CNN으로부터 나온 예측 평점을 이용하여 광고를 중단하고, 이후 Keypoint Matching을 통해 광고를 추천하는 실시간 광고 추천 시스템을 개발하고자 한다.
