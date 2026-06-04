# Changelog

## v39 (2026-06-01) — 최종 제출

### 모델
- v38 구조(MLP 3-seed 앙상블) + **TrajTransformer** 3-seed 앙상블 병행
- 두 모델의 OOF R-Hit 기반 가중 블렌딩 → XGB 메타 게이팅 적용

### TrajTransformer 아키텍처
- `vel_proj` Linear(3 → 128) + 학습 가능 위치 임베딩 (trunc_normal init)
- Pre-LN `TransformerEncoder` (d_model=128, nhead=4, num_layers=4, FFN=512)
- Global Average Pooling → head Linear(135, 256) + GELU + Linear(256, 64) + Linear(64, 3)
- 파라미터 수: ~180K

### TTA (Test Time Augmentation)
- 테스트 추론 시 32회 랜덤 SO(3) 회전 적용 후 원래 프레임으로 역변환해 평균
- 각 회전 Q에 대해: `res_orig = Q.T @ R_Q.T @ (model(Q·traj) × scale)`
- 빠른 속력·급선회 구간의 예측 분산 감소 목적

### 변경 사항
- `TrajTransformer`: 신규 클래스 추가
- `train_transformer_5fold()`: smooth R-Hit loss + TTA 32회 추론 포함 신규 함수
- `main()`: Transformer 3-seed 앙상블 → OOF 기반 MLP:TF 가중치 계산 → 블렌딩 → 게이팅

### 결과
- Dacon Public: **0.6812** (v38 대비 +0.0002↑)
- **Private: 0.6784 / 219등** (최종 순위)

---

## v38 (2026-05-31)

### 모델
- v37 구조에서 Pseudo-label 제거, Multi-seed 앙상블 도입
- MLP(hidden=512) × 3 seed (42/123/456) × 5-Fold + XGB 게이팅

### 변경 사항
- `main()`: SEEDS=[42,123,456], 각 seed별 OOF 누적 후 평균 → 앙상블 OOF
- Pseudo-label Phase 제거 (v34에서 포화 확인됐던 방식)
- N_AUG=9, epochs=150, batch_size=2048 (v37과 동일)
- `train_mlp_5fold()`: `extra_X/extra_y/extra_scale` 파라미터 사용 안 함

### 결과
- Seed별 OOF: s42=0.6621 / s123=0.6656 / s456=0.6602
- 앙상블 OOF: 0.6691, Gated OOF: 0.6702
- Dacon Public: **0.681** (v37 대비 +0.0038↑, 역대 최고)

---

## v37 (2026-05-31)

### 모델
- XGBoost 회귀자 → **TrajMLP** (tabular 피처 기반 MLP) 교체
- smooth R-Hit@1cm loss로 직접 최적화
- cv_smooth 앵커 + Pseudo-label 2-Phase + XGB 게이팅

### TrajMLP 아키텍처
- Input BN → Linear(424, 512) + BN + GELU → Linear(512, 512) + BN + GELU + residual → Linear(512, 256) + BN + GELU → Linear(256, 3)
- Dropout=0.3

### smooth R-Hit Loss
```python
loss = -sigmoid(k × (1 - dist_cm)).mean()   # k=10
```
1cm 임계값 근처에서 미분 가능한 R-Hit 근사. XGBoost MSE 대비 Q5(빠름) R-Hit 0.349→0.464(+0.115) 대폭 개선.

### 결과
- OOF: 0.6624 (Gated 0.6680)
- Dacon Public: **0.6772** (v36 대비 +0.0142↑↑, 역대 최고)

---

## v36 (2026-05-31)

### 주요 변경
- `make_xgb_features()`: 신규 피처 41개 추가 (383 → 424개)
  - 3D 비틀림 torsion (8) + 통계 (4)
  - 예측 불일치도: cv_smooth vs CT (3), cv_smooth vs CA (3), CT vs CA (3)
  - 절대 위치 abs_pos_last, abs_pos_first (6)
  - ω 변화율 d_omega (8) + 통계 (3)
  - 곡률·속력 변동계수 kappa_cv, speed_cv, kappa_last3_std (3)

### 결과
- Dacon Public: **0.6630** (v32 대비 -0.0028↓ — 비틀림·절대위치 노이즈, 불일치도·dω는 유효)

---

## v35 (2026-05-31)

### 모델
- v32 구조(XGBoost + cv_smooth 앵커 + Pseudo-label) + **LightGBM** 앙상블 추가

### 변경 사항
- `train_group()`: `model_type='lgb'` 분기 추가 — LGBMRegressor 동일 하이퍼파라미터
- `main()`: XGBoost Phase2 OOF + LGB Phase2 OOF 50:50 블렌딩 후 게이팅

### 결과
- Phase2 OOF: XGB 0.6152 / LGB 0.6191 → 블렌드 0.6196 (Gated 0.6528)
- Dacon Public: **0.6636** (v32 대비 -0.0022↓)

### 실패 분석
- LGB OOF 수치는 XGB보다 높으나 Dacon에서 일반화 열세 (v14와 동일한 패턴 반복)
- 50:50 블렌딩이 XGBoost 성능 희석
- v32 단독이 여전히 최고

---

## v34 (2026-05-30)

### 모델
- v32 구조 + **Phase 3** 추가 (gated pseudo-label)

### 변경 사항
- `main()`: Phase2 gated 예측 기반 테스트 pseudo-label 재추출 → Phase3 학습
- Phase3 pseudo-label 선별 기준: gate_prob > 0.7 AND 예측 일관성 기준

### 결과
- Phase2 OOF 0.6180 → Phase3 OOF 0.6169 (Gated 0.6508)
- Dacon Public: **0.6658** (v32와 동일 — Phase3 효과 없음)

### 실패 분석
- Phase3 OOF 소폭 하락(-0.0011): 이미 Phase2에서 pseudo-label이 포화 상태
- Gated OOF 미세 상승(+0.0004)은 노이즈 수준
- Pseudo-label 반복 재학습의 한계 확인

---

## v33 (2026-05-30)

### 모델
- v32 구조 + 이진 게이트 → **소프트 회귀 게이트** 실험

### 변경 사항
- 기존: `gate_prob` (XGBClassifier 확률) × residual
- 변경: 최적 혼합 비율 α*를 OOF 기반으로 샘플별 직접 계산
  - `α* = argmin_α ||blend + α·residual - true||`
  - 정답이 있는 학습셋에서 α* 계산 후 XGB 회귀자로 테스트 예측

### 결과
- OOF: 0.6180 (Gated 0.6497 — v32 Gated 0.6504 대비 -0.0007↓)
- Dacon Public: **0.6648** (v32 대비 -0.0010↓)

### 실패 분석
- α* 분포: mean=0.61, std=0.42 → 대부분 0 또는 1에 집중 (사실상 이진)
- R-Hit@1cm는 거리 임계값 지표 → 연속적인 α 최적화보다 이진 게이트가 더 적합
- v32 이진 게이팅이 우세

---

## v32 (2026-05-30)

### 핵심 변경
- **앵커 교체**: CT 블렌드(0.5369) → **cv_smooth(Kalman CA)(0.5812)**
- `predict_ca_kf()`: CA 칼만 필터로 스무딩된 속도에서 CV 예측 (cv_smooth)
- `main()`: blend = cv_smooth_train (CT 블렌드 앵커 완전 폐기)
- Pseudo-label 2-Phase 유지

### 결과
- OOF: 0.6180 (Gated 0.6504)
- Dacon Public: **0.6658** (v31 대비 +0.0128↑↑, 역대 최고)
- 앵커 교체가 단일 최대 개선 요인

---

## v31 (2026-05-30)

### 모델
- v30 구조(Pseudo-label 2-Phase) + **Multi-seed(×5)** 앙상블

### 변경 사항
- `main()`: SEEDS=[42, 7, 123, 456, 999] 5개 seed로 Phase1·Phase2 각각 학습
- 각 seed의 Phase2 OOF 평균 → 게이팅

### 결과
- Phase2 OOF: 0.6241 (Gated 0.6418)
- Dacon Public: **0.6530** (v32 대비 -0.0128↓, v30 대비 -0.0024↓)

### 실패 분석
- OOF는 +0.0035 상승했으나 Dacon에서 -0.0024 하락 → OOF↑-Dacon↓ 패턴 재현
- Multi-seed 앙상블이 게이트 캘리브레이션을 흔들어 soft gate 적용 오류 발생
- seed 다양화로 인한 OOF 분산 감소가 오히려 gate 학습 신호를 왜곡한 것으로 추정

---

## v30 (2026-05-30)

### 모델
- v19 구조(XGBoost + cv_smooth 앵커) + **Pseudo-label 2-Phase** 재학습

### 변경 사항
- `main()`: 2-Phase 구조 도입
  - Phase1: 기본 5-Fold 학습 → OOF per-fold std 계산
  - Phase2: std 기준 하위 40%(고신뢰) 테스트 샘플을 pseudo-label로 추가 → 재학습
- `train_group()`: `extra_X/extra_y` 파라미터 추가, per-fold 테스트 잔차 수집 및 반환

### 결과
- Phase1 OOF 0.6138 → Phase2 OOF 0.6206 (Gated 0.6422)
- Dacon Public: **0.6554** (v19 대비 +0.0002↑, 역대 최고 타이)

### 분석
- pseudo-label로 OOF +0.0068 상승, 테스트 분포에 적응 효과 확인
- 고신뢰 40% 기준: fold간 예측 표준편차 하위 40%

---

## v29 (2026-05-30)

### 모델
- v19 구조 복귀 + **원호 피팅(Circle Fit)** 앵커·피처 추가 시도

### 변경 사항
- `predict_circle_fit()`: PCA 투영 + 2D 최소제곱 원 피팅 함수 추가 (n_pts=6)
- `make_xgb_features()`: `cf_pred_L(3), cf_vs_cv_L(3), cf_vs_ct_L(3), cf_weight(1)` 10개 추가 (383→393 피처)
- CF 앵커 단독 테스트 결과: R-Hit=0.0105 (실패)

### 결과
- OOF: 0.6166 (Gated 0.6385)
- Dacon Public: **0.6508** (v19 대비 -0.0044↓)

### 실패 분석
- 6개 점으로 원호 피팅 시 노이즈에 극히 민감 → CF 앵커 단독 R-Hit=0.0105 (CV 0.5788 대비 사실상 무의미)
- 피처로 추가해도 OOF·Dacon 모두 v19보다 낮음 → 정보 노이즈로 작용

---

## v28 (2026-05-28)

### 모델
- v27 GRU 모델 스케일 확장

### 변경 사항
- `TrajGRU`: hidden 32→**128**, head `Linear(hidden+7, 64)`→`Linear(hidden+7, 128)` (~68K params)
- `train_gru_5fold()`: N_AUG 9→**19** (~200K 증강 샘플)

### 결과
- Dacon Public: **0.6428** (v27 대비 +0.0088↑)
- XGBoost v19(0.6552) 대비 -0.0124↓

### 분석
- 모델 용량 증가는 유효 — hidden=32(5K)는 명백히 과소적합이었음
- 그러나 GRU 구조 자체가 XGBoost 대비 열세
- 10K 데이터 환경에서 explicit 물리 피처 377개 > raw sequence 모델

---

## v27 (2026-05-26)

### 모델
- XGBoost 제거, **TrajGRU** (초소형) 도입 첫 시도

### 변경 사항
- `TrajGRU`: GRU(3→32, 1layer) + head Linear(32+7, 64)+Linear(64, 3) (~5K params)
- `train_gru_5fold()`: 5-Fold + N_AUG=9, 100 epochs, batch=512
- 입력: 속력 기반 정규화 velocity sequence(10,3) + 물리 피처(7)
- 손실: MSELoss

### 결과
- Dacon Public: **0.634** (v19 XGBoost 0.6552 대비 -0.021↓)

### 실패 분석
- hidden=32 → 파라미터 5K: 명백한 과소적합
- 10K 샘플에서 explicit 물리 피처 기반 XGBoost가 sequence 모델보다 강건
- v28에서 hidden=128로 확장

---

## v26 (2026-05-26)

### 모델
- v25 구조(XGBoost + CT 블렌드 앵커) + CT 모델 안정화 + speed-trend 내부 블렌드

### 변경 사항
- `predict_ct()`: ω 단일 프레임 → **마지막 3프레임 평균**으로 안정화
- `_speed_trend_inner()`: 접선 가속도 보정 CV (`cv_trend = cv + 0.5·a_t·dt²`) — trend_w ∈ [0, 0.4]
- `batch_physics_blend()`: speed-trend inner + CT 2-layer 블렌드로 교체

### 결과
- OOF: 0.6160 (Gated 0.6396)
- Dacon Public: **0.6536** (v25 대비 -0.0008↓)

### 실패 분석
- 강선회 OOF +0.025 상승했으나 Dacon에서 하락 → OOF↑-Dacon↓ 패턴 7연속
- ω 3프레임 평균으로 안정화됐지만 앵커 구조 자체의 한계

---

## v25 (2026-05-26)

### 모델
- v24 실패 후 **v19 구조 복귀** + cv_smooth 피처만 보존

### 변경 사항
- `main()`: CA 앵커 제거, CT 블렌드 앵커 복원 (v19 수준)
- `make_xgb_features()`: cv_smooth 관련 6개 피처 유지 (`cv_smooth_L, cv_smooth_vs_raw_L`)

### 결과
- OOF: 0.6148 (Gated 0.6378)
- Dacon Public: **0.6544** (v19 대비 -0.0008↓)

### 분석
- cv_smooth 피처 자체는 유효한 정보를 담고 있음
- 앵커로 쓰는 것(v24)은 실패, 피처로 활용하는 것은 소폭 유효

---

## v24 (2026-05-26)

### 모델
- v19 구조 + **Kalman CA 앵커** 추가 → CV + CT + CA 3-way 블렌드

### 변경 사항
- `predict_ca_kf()`: CA 칼만 필터 앵커 — 접선 가속도 기반 CA 예측 + cv_smooth 출력
- `make_xgb_features()`: ca_kf_pred(3), ca_kf_vs_cv(3), cv_smooth_L(3), cv_smooth_vs_raw_L(3) 10개 추가
- `main()`: blend = CV + CT + CA 3-way 가중 합산

### 결과
- OOF: 0.6138 (Gated 0.6405)
- Dacon Public: **0.6536** (v19 대비 -0.0016↓)

### 실패 분석
- CA-KF 앵커 단독 R-Hit: 0.3968 (CV 0.5788보다 훨씬 낮음) — 앵커로 부적합
- 그러나 cv_smooth 피처(6개)는 XGBoost에서 유효 → v25에서 피처만 유지

---

## v23 (2026-05-26)

### 모델
- v19 구조에서 XGBoost 회귀자 → TrajTransformer로 교체
- 메타 게이팅은 XGBoost 분류기 유지 (피처 377개)

### 아키텍처
- vel_embed(3→64) + 학습 가능 위치 인코딩(10, 64)
- Pre-LN TransformerEncoder (d_model=64, nhead=4, num_layers=2, FFN=256) → last token
- Head: Linear(64+7, 128) + GELU + Dropout + Linear(128, 3)
- 파라미터 수: 110,467
- 입력: 속력 기반 정규화 (vel × DT·HORIZON / disp_scale, ct_pred_L / disp_scale)

### 변경 사항
- `prepare_dl_inputs()`: 로컬 프레임 속도 시퀀스(10,3) + 물리 피처(7) 생성
- `TrajTransformer`: 시퀀스 인코더 + 물리 피처 결합 → 정규화 잔차 예측
- `train_dl_5fold()`: 5-Fold CV + 회전 증강(N_AUG=4) + 50 에폭 Adam+CosineAnnealingLR
- CT 결과를 한 번만 계산해 XGBoost 피처와 DL 입력에 재사용

### 결과
- Fold별 R-Hit: 0.6325 / 0.6240 / 0.6255 / 0.6105 / 0.6200 (분산 높음)
- OOF R-Hit: 0.6225 (+0.009 vs v19 XGBoost 0.6134)
- Gated OOF: 0.6367 (+0.0007 vs v19 0.6360)
- Dacon Public: **0.6472** (v19 대비 -0.008 ↓)

### 실패 분석
- OOF는 개선됐지만 Dacon에서 역전 → Transformer 과적합 또는 일반화 열세
- Fold 분산 (std ≈ 0.007) vs XGBoost v22 (std ≈ 0.003) — DL이 더 불안정
- XGBoost의 명시적 물리 피처 377개가 test 분포를 더 잘 커버하는 것으로 추정
- 강선회(ct_weight 0.6~1.0): R-Hit 0.385 (XGBoost 0.397보다도 낮음)
- v19가 여전히 Dacon 최고 (0.6552)

---

## v22 (2026-05-25)

### 모델
- v21 구조 유지 + TTA (Test Time Augmentation) 추가

### 변경 사항
- `train_group`에 `fold_models` 저장, TTA 파라미터 추가
- 테스트 시 8번 랜덤 회전+스케일 적용 후 예측, 역변환해 원본 프레임으로 복원
- `delta_orig = Q.T @ delta_aug / s` 로 역변환
- 최종: `(원본 예측 + TTA 8회 평균) / 2`

### 결과
- OOF R-Hit: 0.6258 (v21과 동일 — TTA는 테스트에만 적용)
- Dacon Public: 0.6534 (v21 대비 -0.0016 ↓)

### 실패 분석
- 속력 정규화(v17)로 인해 스케일 TTA가 이론적으로 무의미: `s*res/(s*disp) = res/disp`
- 실제 차이는 speed_last·pos_x 등 절대값 피처의 노이즈에서만 발생 → 오히려 예측 흔들림
- 게이트가 원본 예측 분포로 학습됐는데 TTA 평균 후 분포가 달라져 캘리브레이션 불일치
- v19가 여전히 Dacon 최고 (0.6552)

---

## v21 (2026-05-25)

### 모델
- v20 구조 유지 + 속력 스케일 증강 추가

### 변경 사항
- 각 증강 iteration에 SO(3) 회전 후 속력 스케일 s ∈ [0.5, 2.0] 적용
- 마지막 관측점 기준 상대 거리 스케일: `traj_aug = last + (traj_rot - last) * s`
- `disp_scale_aug = disp_scale * s` (속력 정규화 스케일 일치)
- ct_weight 불변 (ω = a_n/speed, 분자·분모 동시 s배 → 상쇄)

### 결과
- OOF R-Hit: 0.6258 (raw) / 0.6433 (gated) — v19 대비 +0.0124 ↑↑
- Dacon Public: 0.655 (v19 대비 ≈ 0)
- OOF-Dacon 갭: 0.0418 → 0.0292 (OOF가 테스트 분포를 더 잘 반영)

### 분석
- Q5(빠름) OOF: 0.341 → 0.382 (+0.041) — 속력 스케일 증강이 Fast 패턴 학습에 효과적
- Q1(직진) OOF: 0.576 → 0.606 (+0.030) — 예상 외 개선
- 강선회: 0.397 → 0.397 — 변화 없음
- ct_extrap_vs_cv 3·4위 진입 — v20 단독 실패했던 외삽 CT가 스케일 증강과 결합 시 유효
- Dacon 미향상 원인: 테스트 속력 분포가 학습과 다르거나 게이트 캘리브레이션 문제

---

## v20 (2026-05-25)

### 모델
- v19 구조 유지 + 다중 CT 앵커 + 각가속도 피처 추가

### 변경 사항
- `_ct_local_from_omega(vel_local, a_n_vec_local, omega)` 헬퍼 추가
- 피처 20개 추가 (377 → 397개)
  - `ct_prev3`: 3스텝 전 ω 기준 CT 예측 (3)
  - `ct_extrap`: 외삽 ω 기준 CT 예측 (3)
  - `ct_prev3_vs_cv`, `ct_extrap_vs_cv`: CT vs CV 차이 (6)
  - `ct_prev3_vs_now`, `ct_extrap_vs_now`: CT 변화량 (6)
  - `domega_dt`: 각가속도 dω/dt (1)
  - `omega_extrap`: 외삽 후 ω 값 (1)

### 결과
- Dacon Public: 0.6542 (v19 대비 -0.0010 ↓)

### 실패 분석
- dω/dt를 3스텝(120ms) 차분으로 추정 → 모기 불규칙 운동에서 노이즈가 신호를 압도
- ct_prev3의 방향 벡터를 현재 a_n으로 근사 → 실제 과거 선회 방향과 불일치
- omega 시계열이 기존 피처에 이미 충분히 담겨 있어 중복 정보
- 강선회 문제는 CT 개선만으로 해결하기 어려움 — 10개 관측점으로 ω 변화 추정에 근본적 한계

---

## v19 (2026-05-25)

### 모델
- v17 구조(XGBoost + CT 블렌드 + 3D 회전 증강 + 5-Fold + 속력 정규화) 유지
- **메타 게이팅(Meta Gating)** 추가: XGBClassifier가 "XGBoost 보정이 도움이 되는가" 분류

### 핵심 변경
- `gate_labels = (oof_err < blend_err)` — OOF 오차 < 블렌드 오차이면 1 (개선), 아니면 0 (악화)
- XGBClassifier(depth=4, 300 trees, lr=0.05)로 5-Fold OOF 기반 게이트 학습
- Soft gate 적용: `final = blend + gate_prob × xgb_residual`
  - gate_prob=1.0 → XGBoost 전적으로 신뢰, gate_prob=0.0 → 블렌드 앵커 유지
- OOF-gated (in-sample): `oof_gated = blend + gate_prob_train × (oof_preds - blend)`
- 테스트: gate_prob_test는 train과 동일한 gate_clf로 예측 (누수 없음)

### 결과
- OOF R-Hit: 0.6134 (raw, v17과 동일)
- OOF-Gated R-Hit: 0.6360 (+0.0226 over raw, in-sample)
- Dacon Public: **0.6552** (v17 대비 +0.0074 ↑↑, 역대 최고)
- 1등(0.7)과의 격차: 0.0448

### 게이트 분석
- 개선 샘플: 6,241개 (62.4%), 악화 샘플: 3,759개 (37.6%)
- OOF-Gated(0.6360) < Dacon(0.6552) → 테스트에서 더 잘 일반화
- 게이트가 XGBoost 보정이 오히려 노이즈인 샘플을 차단해 실제 환경에서 효과 극대화

### 잔존 문제
- Q5(빠름): R-Hit=0.341 (여전히 하위)
- 강선회(ct_w 0.6~1.0): R-Hit=0.397
- CV-last 대비 악화 샘플: 3,575개 (35.8%) — 게이팅 후에도 남음

---

## v18 (2026-05-25)

### 모델
- 속력 60th percentile 기준으로 Slow(Q1~Q3) / Fast(Q4~Q5) 그룹 분리
- 각 그룹 독립 증강 + 5-Fold 학습 (`train_group` 헬퍼 함수)
- Slow: N_AUG=4 (6,000개 → 30,000개), Fast: N_AUG=6 (4,000개 → 28,000개)
- 테스트: 속력 기준 모델 라우팅

### 결과
- Slow-OOF: R-Hit=0.7395 / Fast-OOF: R-Hit=0.4002
- 전체 OOF: 0.6038 (v17 대비 -0.0096 ↓)
- Dacon Public: 0.6454 (v17 대비 -0.0024 ↓)

### 실패 분석
- Fast 모델 4,000개 샘플 → 데이터 부족으로 일반화 실패 (Q4 0.499→0.464)
- 분리의 이점(전문화) < 데이터 감소의 손실
- 단일 10K 모델이 slow/fast 패턴을 함께 학습하는 것이 더 유리
- v17(단일 모델 + 속력 정규화)이 여전히 역대 최고

---

## v17 (2026-05-24)

### 모델
- XGBoost + CT 블렌드 + 3D 회전 증강 + 5-Fold (v15 구조 유지)
- **속력 기반 잔차 정규화** 추가

### 핵심 변경
- 학습 타깃: `res_local` → `res_local / (speed_last × dt)` (무차원 비율)
- 예측 후 역정규화: `pred_norm × (speed_last × dt)` 로 복원
- 증강 샘플: 속력 크기는 회전에 불변 → `disp_scale` 그대로 재사용
- 동기: 빠른 샘플(잔차 크기 大)과 느린 샘플(잔차 크기 小)의 학습 신호 스케일 통일

### 결과
- OOF R-Hit: 0.6134 (v15와 동일)
- Dacon Public: 0.6478 (v15 대비 +0.0022 ↑, 역대 최고)
- 1등(0.7)과의 격차: 0.0522

### 분석
- OOF는 동일하나 Public이 향상 → 정규화가 OOF보다 테스트 일반화에 더 효과적
- Q5(빠름) OOF: 0.347 → 0.341 (소폭 하락) — 속력 자체의 문제는 여전히 남음
- 강선회 OOF: 0.388 → 0.397 (+0.009) — 소폭 개선
- 피처 중요도 변화: turn_last·omega 계열이 상위 부상 (방향 피처가 더 discriminative)
- 악화 샘플: 3,620 → 3,575 (45개 감소)

---

## v16 (2026-05-24)

### 모델
- v15와 동일 (분석 전용, 미제출)

### 변경 사항
- OOF 에러 분석 코드 추가 (`run_error_analysis`)
- 속력·곡률·CT weight·속력추세별 R-Hit 분해
- `output/oof_analysis_v16.csv` 저장

### 주요 발견
- Q5(빠름): R-Hit=0.347 — 속력이 가장 큰 실패 변수
- 강선회(ct_w 0.6~1.0): R-Hit=0.388 — 두 번째 문제
- CV-last 대비 악화 샘플: 3,620개 (36.2%) — 보정이 오히려 노이즈
- 가속 구간: R-Hit=0.518 — 감속(0.648)·등속(0.674)보다 유의미하게 낮음

---

## v15 (2026-05-23)

### 모델
- XGBoost + CT 블렌드 앵커 + 3D 회전 증강 + 5-Fold 앙상블

### 변경 사항
- 3D 회전 증강(N_AUG=4): 학습 데이터 10,000개 → 50,000개
  - 랜덤 SO(3) 회전(QR 분해)으로 궤적 전체 회전 후 피처 재계산
  - ct_weight는 회전 불변(ω = a_n/speed, 크기만 사용) → 그대로 재사용
- 누수 방지 Fold 설계: val = 원본 인덱스만, train = 원본 + 증강 4벌 (val 원본 제외)

### 결과
- OOF R-Hit: 0.6134 (Fold: 0.6205 / 0.6140 / 0.6160 / 0.6090 / 0.6075)
- Dacon Public: 0.6456 (v13 대비 +0.0036 ↑, 역대 최고)
- 1등(0.7)과의 격차: 0.0544

### 피처 중요도 변화
- 1위: ct_pred_y (v13에서도 상위권 → CT 피처 여전히 핵심)
- 2위: a_t_t9 (접선 가속도 마지막 타임스텝)
- 3위: ct_pred_z
- 4위: turn_last

---

## v14 (2026-05-22)

### 모델
- XGBoost 5-Fold + LightGBM 5-Fold 50:50 앙상블

### 결과
- OOF: XGB 0.6030 / LGB 0.5960 / Ensemble 0.6064
- Dacon Public: 0.6406 (v13 대비 -0.0014 ↓)

### 분석
- LightGBM이 동일 하이퍼파라미터에서 XGBoost보다 약함 (OOF -0.007)
- 50:50 블렌드가 XGBoost 성능을 희석 → public도 하락
- 피처 중요도 합산 오류: XGB는 gain 비율(0~1), LGB는 split 횟수(정수) → 스케일 불일치로 의미 없는 수치 출력
- v13(XGBoost 단독 5-Fold) 이 여전히 역대 최고

---

## v13 (2026-05-22)

### 모델
- XGBoost + CT 블렌드 앵커 (v12 구조 유지) + 5-Fold 앙상블

### 변경 사항
- 단일 8:2 분할 → 5-Fold Cross Validation
- 학습 데이터: 8,000개 → 10,000개 전체 활용
- 테스트 예측: 단일 모델 → 5개 모델 잔차 평균
- 검증 지표: val R-Hit → OOF R-Hit (더 신뢰도 높음)

### 결과
- OOF R-Hit: 0.6030 (Fold 1~5: 0.608 / 0.610 / 0.597 / 0.604 / 0.597)
- Dacon Public: 0.6420 (v12 대비 +0.0086 ↑↑, 역대 최고)
- 1등(0.7)과의 격차: 0.058 → 계속 좁혀지는 중

---

## v12 (2026-05-22)

### 모델
- XGBoost + CT(Constant Turn Rate) 물리 모델 블렌드 앵커
- 잔차 기준: `true - cv_last` → `true - physics_blend`

### CT 모델 원리
- 현재 각속도 ω = a_n / speed (rad/s)로 원호 경로 예측
- 선회각 theta = ω × 0.08s 기반 CT 신뢰도(ct_weight) 계산
- `blend = (1 - ct_weight) × cv_pred + ct_weight × ct_pred`
- 직진 중 → CT weight ≈ 0 (CV 그대로), 강한 선회 → CT weight → 1

### 피처 변경 (364 → 377개, +13)
- ct_pred_L (3): CT 예측 로컬 변위
- ct_vs_cv_L (3): CT - CV 차이 (선회 보정량)
- ct_weight (1): 선회 강도 기반 CT 신뢰도
- (기타 내부 계산 피처 6개 추가)

### 주요 발견
- CT-blend 앵커 자체 R-Hit: 0.5385 (CV 0.5788보다 낮음)
- 그러나 CT 피처 중요도 2위(ct_pred_z), 3위(ct_pred_y), 4위(ct_vs_cv_z)
- 앵커가 나빠도 XGBoost가 CT 방향 정보를 활용해 더 정확히 보정
- CT weight 평균 0.371 → 데이터의 약 37%가 의미 있는 선회 구간

### 결과
- Dacon Public: 0.6334 (v10 대비 +0.0018 ↑, 역대 최고)

---

## v11 (2026-05-22)

### 모델
- XGBoost + 잔차학습 (v10 구조 유지)

### 피처 변경 (306 → 364개, +58)
- 각속도 ω = a_n/speed 시계열 (9) + 통계 mean/std/last/trend (4)
- 등가속도 예측 ca_pred_L = vel[-1]·dt + 0.5·acc[-1]·dt² (3) + vs cv_L 차이 (3)
- 부호 있는 평면별 곡률 kappa_xy_s, kappa_xz_s 시계열 (9×2) + 통계 (3×2)
- 속도 방향 비율 vel_lat_frac(vy/v), vel_vert_frac(vz/v) 통계 (3×2)
- 다항식 피팅 RMSE quad_rmse, cubic_rmse 각 축 (3×2)
- 다항식 vs cv_3 비교 quad_L − cv_3_L, cubic_L − cv_3_L (3×2)
- 선회 방향 일관성 turn_dir_y, turn_dir_z, a_n_ratio (3)

### 결과
- Dacon Public: 0.6286 (v10 대비 -0.003 ↓)
- FE 포화 확인 — 피처 추가만으로는 한계, 방향 전환 필요

---

## v10 (2026-05-18)

### 모델
- XGBoost + 잔차학습 (v9 구조 유지)

### 피처 변경 (274 → 306개, +32)
- 법선 가속도 a_n, 접선 가속도 a_t 시계열 (9×2) + 통계 mean/std/recent_mean/last (6)
- 다항식 외삽: quadratic/cubic fit to 11 positions (3×2) + vs cv_L 차이 (3×2)
- 평면별 곡률(크기) kappa_xy, kappa_xz 시계열 (9×2) + 통계 (4×2)
- 속도 R²(선형 적합도) vel_r2 (3), 가속도 추세 acc_slope (3)
- vel_last3_vs_last7, vel_last2_vs_last5 등 멀티스케일 속도 비교

### 결과
- Dacon Public: 0.6316 (v9 대비 +0.0070 ↑↑, 역대 최고)

---

## v9 (2026-05-18)

### 모델
- XGBoost + 잔차학습 (v7 하이퍼파라미터 복구)
- 자기중심 로컬 좌표계 적용: 마지막 속도벡터 → +x축 정렬

### 주요 변경
- `_local_frame_rotation(vel)`: 3×3 회전행렬 R 생성
- 모든 벡터 피처(속도·가속도·저크·위치·CV delta 등)를 로컬 프레임으로 변환
- 잔차 타깃도 로컬 프레임에서 학습, 예측 후 역회전으로 글로벌 복원
- cv_align: cv_all vs cv_last 정렬도로 교체 (기존 항등식 버그 수정)

### 결과
- Dacon Public: 0.6246 (v7 대비 +0.0006)

---

## v8 (2026-05-18)

### 모델
- XGBoost + 잔차학습
- 과정규화: max_depth 6→5, min_child_weight 3→5

### 피처 변경 (274 → 238개)
- 중요도 하위 50개 피처 제거, 신규 15개 추가

### 결과
- Dacon Public: 0.6228 (v7 대비 -0.0012 ↓)
- 과정규화로 하락 → v9에서 하이퍼파라미터 원복

---

## v7 (2026-05-18)

### 모델
- XGBoost + 잔차학습 (v3 구조 복귀, 피처 대폭 확장)
- colsample_bytree: 0.8 → 0.7 (피처 증가에 따른 정규화)

### 피처 변경 (94 → 274개)
- 저크(jerk, 3차 미분) 시계열 및 통계
- Frenet 곡률(κ = |v×a|/|v|³) 시계열 및 통계
- 회전축(rot_axes) 시계열
- 멀티스케일 CV (1/3/5/전체 스텝 평균, 스케일 간 분산)
- 경로 직선성(straightness = displacement / path_length)
- 속력 변화율(speed_delta) 시계열 및 통계
- 위치 시계열 전체(11스텝)

### 결과
- Dacon Public: 0.6240 (역대 최고, v3 대비 +0.0150)

---

## v6 (2026-05-18)

### 모델
- BiLSTM + 잔차학습 (v5 구조 유지)

### 피처 변경
- `make_seq_features`: turn_cos 추가 → (10, 9)
- `make_global_features`: 18 → 29개
  - turn_cos 통계 (mean/std/min/last) — 방향 전환 패턴
  - 속력 변화율 통계 (mean/std) — 가감속 추세
  - 마지막 속도 단위벡터 (3) — 방향 정보 명시
  - CV-delta vs 마지막 속도 정렬도 (1)
  - 속력 선형 기울기 (1)

### 결과
- Dacon Public: 0.6044 (v5 대비 +0.0012, v3 XGBoost 0.6090 미달)

---

## v5 (2026-05-17)

### 모델
- v4 BiLSTM 구조 유지 + 글로벌 피처 확장

### 변경 사항
- `make_global_features()`: 15 → 18개 (turn_cos 통계, 속력 기울기 추가)
- 시퀀스 입력 정규화 추가 시도

### 결과
- Dacon Public: **0.6032** (v4 대비 -0.0006↓)

### 실패 분석
- FE 강화에도 BiLSTM이 XGBoost v3(0.6090) 대비 열세 지속
- 10K 샘플 환경에서 시퀀스 모델의 한계 확인 → v6에서 추가 피처 강화 후 포기

---

## v4 (2026-05-17)

### 모델
- XGBoost 잔차학습(v3) → **LSTM 잔차학습** 교체 시도

### 변경 사항
- `TrajectoryLSTM`: LSTM(hidden=128, 2layer) + FC(128→3)
- 입력: 속도 시계열(10, 3) + 글로벌 피처(15개) concat
- 타깃: v3와 동일 (true_xyz - cv_last)
- Adam optimizer, MSELoss, 50 epochs

### 결과
- Dacon Public: **0.6038** (v3 XGBoost 0.6090 대비 -0.0052↓)

### 실패 분석
- 10,000개 샘플에서 XGBoost가 LSTM보다 강건
- 명시적 물리 피처(속력·곡률·회전 정보)를 XGBoost가 더 효율적으로 활용
- LSTM은 raw 시퀀스에서 이 정보를 스스로 추출해야 하므로 학습 데이터가 적을 때 불리

---

## v3 (2026-05-17)

### 모델
- XGBoost + 잔차 학습 (Residual Learning)
- 타깃: `true_xyz - cv_last_pred` (CV-last 대비 오차)
- 최종 예측: `cv_last_pred + xgb_residual`

### 피처 변경 (v1 대비)
- CV-last 예측 delta 추가 (`cv_pred - traj[-1]`) → 보정 방향 앵커
- 절대 위치 복구 (v2에서 제거했다가 성능 하락 확인 후 복구)

### 하이퍼파라미터
- v1과 동일 (n_estimators=500, max_depth=6, learning_rate=0.05)

---

## v2 (2026-05-17)

### 변경
- 절대 좌표 피처 제거, max_depth 6→4, n_estimators 500→300
- reg_alpha=0.1, reg_lambda=2.0 정규화 추가
- 결과: 0.5950 (v1 대비 하락) → 폐기

---

## v1 (2026-05-17)

### 모델
- XGBoost (MultiOutputRegressor, 각 축 독립 예측)
- 타깃: 절대 좌표 대신 t=0 대비 변화량(delta) 예측

### 피처 (94개)
- 속도 시계열 전체 (10스텝 × 3축)
- 가속도 시계열 전체 (9스텝 × 3축)
- 최근 속도 vs 전체 평균 (방향 전환 감지)
- 속력 통계 (평균, 표준편차, 최근값)
- t=0 위치, 전체 변위

### 하이퍼파라미터
- n_estimators=500, max_depth=6, learning_rate=0.05
- subsample=0.8, colsample_bytree=0.8, min_child_weight=3
