# 2021년 2학기 산업인공지능 텀프로젝트

## Script

1. upzip.py
   
    - 단위별로 압축되어 있는 파일 모두 풀어, 한 곳에 저장하기


2. concat.py

    - 압축 푼 파일을 하나의 파일로 합치고, 저장하기


3. time_unit.py

   - 10분 단위로 데이터 변환
   - 데이터는 평균 값으로 치환됨

4. clustering_paper.py

   - 2020년 출판된 논문 '앙상블 기법을 이용한 선박 메인엔진 빅데이터의 이상치 탐지' (DOI: 10.3796/KSFOT.2020.56.4.384) 를 구현
   - autoencoder의 경우, 레이어 사이즈 조정이 안되 에러 발생함. 이에 대한 수정 혹은 pyod 패키지 내 torch 기반의 모델로 변환 필요

5. LSTM_autoencoder.py

   - 다음 사이트를 참조하여 만들어진 모델(https://www.kaggle.com/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders)
   - 일정 기간(ts, time steps) 내 데이터의 특징값을 추출하는 것을 목표로 함
   - autoencoder의 특징 문헌정리 필요

6. timewindow_labeling.py
   
   - 시간 단위로 분류된 outlier 결과를 time window 단위로 변환
   - 기존 outlier는 0 또는 1의 값을 가짐
   - time window 단위의 score는 기간 내 outlier 비율(평균)로 결정됨
   - 이에 따라 index는 0~1사이의 값(확률값으로 볼 수 있음)을 가짐

