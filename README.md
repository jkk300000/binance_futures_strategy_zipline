# 바이낸스 선물 거래 전략 

### 사용 툴 : python 3.11, ta-lib, zipline, scikit-learn

### 1. 기술적 지표 (rsi 등)를 랜덤 포레스트(ml)에 학습시켜 다음 캔들 예측(양봉 or 음봉)
### 2. 기술적 지표 (rsi 등)를 xgboost(ml)에 학습시켜 다음 캔들 예측(양봉 or 음봉)
### 3. zipline 라이브러리를 통해 백테스팅(mdd, sharp ratio 등) 
