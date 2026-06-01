# 모기 비행 궤적 예측 AI

월간 데이콘 - 모기 비행 궤적 예측 AI 경진대회 제출 코드

**최종 결과: Public 0.6812 / Private 0.6784 / 219등**

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

## 최종 파이프라인 (v39)

```
관측 궤적 (11개 위치, 3D)
        │
        ├─── Kalman CA 스무딩 → cv_smooth (앵커)
        │
        ├─── 424개 tabular 피처 생성
        │      └─ 속도/가속도/저크 시계열, CT 물리 모델,
        │         법선/접선 가속도, 곡률, 다항식 외삽, ...
        │
        ├─── [A] TrajMLP (3 seed × 5-Fold)
        │      · Input: tabular 피처 424개
        │      · hidden=512, smooth R-Hit@1cm loss
        │      · N_AUG=9 3D 회전 증강
        │
        ├─── [B] TrajTransformer (3 seed × 5-Fold + TTA×32)
        │      · Input: 속도 시퀀스 (10, 3) + 물리 피처 (7)
        │      · d_model=128, 4-layer, global avg pooling
        │      · 추론 시 32회 랜덤 회전 TTA → 평균
        │
        ├─── OOF R-Hit 기반 MLP:TF 가중 블렌딩
        │
        └─── XGBoost 메타 게이팅
               · "이 샘플에서 보정이 도움이 되는가" 분류
               · final = cv_smooth + gate_prob × ensemble_residual
```

## 주요 설계 선택

| 항목 | 선택 | 이유 |
|------|------|------|
| 앵커 | Kalman CA cv_smooth | CT 블렌드(0.537)보다 높은 0.581, 단일 최대 개선 |
| Loss | smooth R-Hit@1cm | MSE 대비 Q5(빠름) +0.115 개선 |
| 증강 | SO(3) 랜덤 회전 | 방향 불변성 강제, 10K→100K 샘플 |
| TTA | 32회 회전 평균 | 빠른 속력·급선회 구간 분산 감소 |
| 게이팅 | XGBClassifier | 보정이 노이즈인 37% 샘플 차단 |

## 취약 구간 (최종 v39 기준)

| 구간 | R-Hit | 원인 |
|------|-------|------|
| Q5 빠른 속력 | ~0.46 | 절대 변위 크고 방향 민감도 높음 |
| 강선회 (CT w>0.6) | ~0.50 | 10프레임으로 ω 변화 추정 한계 |
| 가속 구간 | ~0.60 | 속력 변화 예측 어려움 |

## 실행

### Docker (권장)

```bash
docker compose up --build
```

### 로컬

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python mosquito_trajectory_prediction.py
```

## 출력

| 파일 | 설명 |
|------|------|
| `output/submission_mlp_v39.csv` | 최종 제출 파일 |
| `output/oof_analysis_v39.csv` | OOF 에러 분석 (속력/곡률/선회 구간별) |
| `logs/v39_log.txt` | 학습 로그 |

## 환경

- Python 3.11
- NVIDIA GB10 (Grace-Blackwell, sm_121), CUDA 13.0
- PyTorch cu128 (Blackwell sm_121 지원)
- xgboost, lightgbm, scikit-learn, numpy, pandas

## 버전 이력 요약

| 구간 | 점수 범위 | 핵심 변화 |
|------|-----------|-----------|
| v1~v6 | 0.599~0.604 | XGBoost / BiLSTM 기초 |
| v7~v13 | 0.624~0.642 | 피처 확장, 5-Fold CV |
| v15~v19 | 0.646~0.655 | 3D 증강, 속력 정규화, 게이팅 |
| v32 | 0.666 | cv_smooth 앵커 교체 (단일 최대 도약) |
| v37 | 0.677 | MLP + smooth R-Hit loss |
| v38~v39 | 0.681~0.6812 | Multi-seed, Transformer TTA |
