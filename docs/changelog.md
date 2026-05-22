# Changelog

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
