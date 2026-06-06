# 모기 비행 궤적 예측 AI

> 월간 데이콘 - 모기 비행 궤적 예측 AI 경진대회 (2026.05 ~ 2026.06)

## 최종 결과

| Public Score | Private Score | 순위 |
|:---:|:---:|:---:|
| **0.6812** | **0.6784** | **219등** |

베이스라인(CV-last) 0.5788 → 최종 0.6812 **(+0.102)**

---

## 문제 정의

LiDAR 센서로 40ms 간격 측정한 모기 3D 좌표 **11개 시점**으로 **+80ms 후 위치**를 예측한다.

```
입력: (x, y, z) × 11개 시점  →  출력: (x, y, z) at t+80ms
평가: R-Hit@1cm — 예측 오차 ≤ 1cm 인 샘플의 비율
```

**핵심 난이도:** 모기 속도 0.3~1 m/s → 80ms 이동거리 최대 8cm. 1cm 이내 적중이 목표.

---

## 접근 방법

### 전체 파이프라인

```
관측 궤적 (11개 위치, 3D)
        │
        ├── Kalman CA 필터 ──────────────── cv_smooth 앵커 (R-Hit 0.581)
        │                                         │
        ├── 피처 엔지니어링 (424개)                │
        │    속도/가속도/저크 시계열                │
        │    CT 물리 모델, 법선·접선 가속도         │
        │    곡률, 다항식 외삽, 불일치도 등         │
        │         │                               │
        │    ┌────┴─────┐                         │
        │    ▼          ▼                         │
        │  TrajMLP  TrajTransformer               │
        │  (tabular) (raw sequence + TTA×32)      │
        │    └────┬─────┘                         │
        │         │ OOF 기반 가중 블렌딩            │
        │         ▼                               │
        │    앙상블 잔차 예측                        │
        │         │                               │
        └─── XGBClassifier 메타 게이팅 ────────────┘
                  │
             gate_prob × 잔차 + cv_smooth = 최종 예측
```

### 핵심 아이디어 5가지

**1. 앵커 기반 잔차 학습**

절대 좌표 대신 물리 모델(앵커)과의 오차(잔차)를 예측한다. 앵커가 좋을수록 잔차가 작아져 학습이 쉬워진다.

```
학습 타깃 = true_xyz - anchor
최종 예측 = anchor + 모델이 예측한 잔차
```

앵커를 CT 블렌드(0.537) → **Kalman cv_smooth(0.581)** 로 교체하는 것만으로 **+0.013** (대회 단일 최대 개선).

**2. smooth R-Hit@1cm Loss**

R-Hit은 1cm 경계의 계단 함수라 역전파가 안 된다. sigmoid로 미분 가능하게 근사해 직접 최적화한다.

```python
loss = -sigmoid(k * (1.0 - dist_cm)).mean()   # k=10
```

MSE 대비 빠른 속력 구간 R-Hit **0.349 → 0.464 (+0.115)**.

**3. SO(3) 3D 회전 증강**

모기 궤적 예측은 방향에 무관해야 한다. 랜덤 회전행렬(QR 분해)로 궤적 전체를 회전시켜 **10K → 100K 샘플** 확장 및 방향 불변성 학습.

```python
Q = random_rotation()
traj_rot  = (Q @ traj.T).T    # 궤적 회전
true_rot  = Q @ true_xyz      # 정답도 동일하게 회전
```

**4. XGBoost 메타 게이팅**

모델 보정이 오히려 노이즈가 되는 샘플(~35%)이 존재한다. OOF 기반으로 "이 샘플에서 보정이 도움이 되는가"를 분류해 선택적으로 적용한다.

```python
gate_labels = (oof_err < anchor_err)     # OOF 기반, 누수 없음
gate_prob   = XGBClassifier.predict(X)   # 0~1 소프트 게이트

final = anchor + gate_prob × residual    # 확신할수록 많이 반영
```

**5. TrajTransformer + TTA**

raw velocity sequence를 직접 인코딩하는 Transformer. tabular 피처 기반 MLP와 상호 보완적으로 앙상블한다. 추론 시 32회 랜덤 회전 TTA로 예측 분산을 감소시킨다.

---

## 모델 아키텍처

### TrajMLP

```
Input BN (424)
  → Linear(424, 512) + BN + GELU + Dropout(0.3)
  → Linear(512, 512) + BN + GELU + Dropout(0.3)  [+ residual]
  → Linear(512, 256) + BN + GELU + Dropout(0.3)
  → Linear(256, 3)
```

- 입력: 사람이 설계한 424개 물리 피처
- 손실: smooth R-Hit@1cm
- 학습: AdamW + CosineAnnealingLR, best checkpoint

### TrajTransformer

```
vel_seq (10, 3) → Linear(3, 128) + 위치 임베딩
  → TransformerEncoder ×4 (d_model=128, nhead=4, Pre-LN)
  → Global Average Pooling
  → concat phys(7) → Linear(135, 256) → Linear(256, 64) → Linear(64, 3)
```

- 입력: raw velocity sequence + CT 물리 피처 7개
- 추론: TTA×32 (랜덤 회전 → 예측 → 역변환 → 평균)

### 앙상블 전략

```python
# OOF R-Hit 기반 가중치
w_mlp = mlp_oof_rhit / (mlp_oof_rhit + tf_oof_rhit)

# 3 seed × 5 fold = 15개 모델 각각
ens_test = w_mlp × mlp_test_res + w_tf × tf_test_res
```

---

## 피처 엔지니어링 (424개)

| 그룹 | 수 | 예시 |
|------|-----|------|
| 기본 시계열 (vel/acc/jerk/speed/kappa) | 168 | `vel_x_t9`, `kappa_t8` |
| 멀티스케일 CV 예측 | 21 | `cv_3_delta_x`, `cv_spread_z` |
| 법선·접선 가속도 | 24 | `a_n_t9`, `a_t_mean` |
| 다항식 외삽 | 12 | `cubic_delta_x`, `quad_vs_cv_z` |
| 평면별 곡률 | 50 | `kappa_xy_t5`, `kappa_xz_s_t9` |
| 각속도 ω | 13 | `omega_t9`, `omega_trend` |
| 요약 통계 | 50 | `straightness`, `speed_ratio` |
| CT·CA 물리 앵커 비교 | 19 | `ct_pred_x`, `cv_smooth_x`, `ct_weight` |
| 예측 불일치도·변동계수 | 12 | `disagree_smooth_ct_x`, `kappa_cv` |
| 3D 비틀림·ω 변화율 | 23 | `torsion_t5`, `d_omega_t7` |
| 절대 위치 | 6 | `abs_pos_last_x` |

---

## 성능 개선 이력

| 버전 | Public | 핵심 변경 | 개선 |
|------|--------|-----------|------|
| CV-last (baseline) | 0.579 | 등속 외삽 | — |
| v7 | 0.624 | 피처 94→274개 확장 | +0.045 |
| v13 | 0.642 | 5-Fold CV 도입 | +0.018 |
| v15 | 0.646 | 3D 회전 증강 (N_AUG=4) | +0.004 |
| v19 | 0.655 | XGBoost 메타 게이팅 | +0.009 |
| v32 | 0.666 | **cv_smooth 앵커 교체** | **+0.011** |
| v37 | 0.677 | **MLP + smooth R-Hit loss** | **+0.011** |
| v38 | 0.681 | Multi-seed 앙상블 (×3) | +0.004 |
| v39 | **0.681** | Transformer + TTA×32 | +0.0002 |

---

## 주요 교훈

- **앵커 품질이 전체를 결정한다** — 잔차 학습에서 기준점의 정확도가 모든 것에 영향을 미친다
- **손실함수를 평가 지표에 맞춰라** — smooth R-Hit이 MSE보다 빠른 속력 구간에서 +0.115
- **10K 샘플에서는 explicit 피처 > raw sequence** — 명시적 물리 피처 424개가 Transformer raw sequence보다 일관되게 우세
- **OOF만 보지 말고 리더보드를 함께 봐야 한다** — OOF↑-Dacon↓ 패턴이 7연속 발생, 앵커 교체 후 해소

---

## 실행

```bash
# Docker (권장)
docker compose up --build

# 로컬
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python mosquito_trajectory_prediction.py
```

**출력 파일**

| 파일 | 설명 |
|------|------|
| `output/submission_mlp_v39.csv` | 최종 제출 파일 |
| `output/oof_analysis_v39.csv` | OOF 에러 분석 |
| `logs/v39_log.txt` | 학습 로그 |

---

## 기술 스택

`Python 3.11` `PyTorch` `XGBoost` `scikit-learn` `NumPy` `pandas`  
`NVIDIA GB10 (Grace-Blackwell)` `CUDA 13.0` `Docker`
