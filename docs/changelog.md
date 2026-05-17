# Changelog

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
