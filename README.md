# 모기 비행 궤적 예측 AI

월간 데이콘 - 모기 비행 궤적 예측 AI 경진대회 제출 코드

## 문제 정의

40ms 간격으로 관측된 11개 시점의 모기 3D 좌표(x, y, z)를 바탕으로  
마지막 관측 시점 기준 **+80ms 이후의 위치**를 예측한다.

- 평가 지표: R-Hit@1cm (예측 오차 ≤ 1cm 비율)
- 좌표계: LiDAR sensor-local (x: forward, y: left, z: up, 단위: m)

## 데이터 구조

```
data/
  train/               # TRAIN_00001.csv ~ TRAIN_10000.csv
  test/                # TEST_00001.csv ~ TEST_10000.csv
  train_labels.csv     # 정답 (id, x, y, z)
  sample_submission.csv
```

각 CSV 컬럼: `timestep_ms, x, y, z` (11행, -400ms ~ 0ms)

## 접근 방법

XGBoost 기반 회귀 모델. 절대 좌표 대신 **t=0 대비 변화량(delta)** 을 예측해  
train/test 환경 차이에 강건하도록 설계.

**입력 피처** (~96개)
- 속도 시계열 전체 (10스텝 × 3축)
- 가속도 시계열 전체 (9스텝 × 3축)
- 최근 속도 vs 전체 평균 (방향 전환 감지)
- 속력 통계 (평균, 표준편차, 최근값)

**출력**: Δx, Δy, Δz → 마지막 관측 위치에 더해 최종 예측

## 실행

### Docker (권장)

```bash
docker-compose up --build
```

### 로컬

```bash
pip install -r requirements.txt
python mosquito_trajectory_prediction.py
```

## 출력

| 파일 | 설명 |
|------|------|
| `submission_xgb.csv` | 제출용 예측 결과 |

## 환경

- Python 3.11
- xgboost, scikit-learn, numpy, pandas, matplotlib
