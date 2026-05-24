# Changelog

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
