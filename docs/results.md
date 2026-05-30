# Results

| 버전 | 날짜 | 모델 | Train R-Hit | Dacon Public Score | 비고 |
|------|------|------|-------------|-------------------|------|
| v1 | 2026-05-17 | XGBoost | 0.9405 | **0.5996** | 절대 좌표 피처 포함, 과적합 의심 |
| v2 | 2026-05-17 | XGBoost | - | **0.5950** | 절대 좌표 제거 + 정규화 강화, 오히려 하락 |
| v3 | 2026-05-17 | XGBoost + 잔차학습 | - | **0.6090** | CV-last 잔차 학습, v1 대비 +0.0094 ↑ |
| v4 | 2026-05-17 | LSTM + 잔차학습 | - | **0.6038** | v3보다 낮음, XGBoost가 우세 |
| v5 | 2026-05-17 | BiLSTM + FE 강화 | - | **0.6032** | v4보다도 낮음, FE 강화 효과 없음 |
| v6 | 2026-05-18 | BiLSTM + FE 강화 v2 | - | **0.6044** | turn_cos·속력기울기 등 추가, v5 대비 +0.0012 ↑ |
| v7 | 2026-05-18 | XGBoost + 피처 확장 | - | **0.6240** | 274개 피처, v3 대비 +0.0150 ↑↑ |
| v8 | 2026-05-18 | XGBoost + 피처 정리 | - | **0.6228** | 과정규화(max_depth 5, mcw 5)로 하락 |
| v9 | 2026-05-18 | XGBoost + 로컬 좌표계 | - | **0.6246** | 자기중심 좌표계 변환, v7 대비 +0.0006 |
| v10 | 2026-05-18 | XGBoost + FE 전면 확장 | - | **0.6316** | 법선/접선 가속도·다항식 외삽·평면 곡률 등, v9 대비 +0.0070 ↑↑ |
| v11 | 2026-05-22 | XGBoost + ω·부호곡률 등 추가 | - | **0.6286** | 364개 피처, v10 대비 -0.003 ↓ (FE 포화) |
| v12 | 2026-05-22 | XGBoost + CT 물리 모델 블렌드 앵커 | - | **0.6334** | CT 선회 모델, v10 대비 +0.0018 ↑ (역대 최고) |
| v13 | 2026-05-22 | XGBoost + CT 블렌드 + 5-Fold 앙상블 | OOF 0.6030 | **0.6420** | K-Fold 전체 데이터 활용, v12 대비 +0.0086 ↑↑ |
| v14 | 2026-05-22 | XGBoost + LightGBM 5-Fold 앙상블 | OOF 0.6064 | **0.6406** | LGB OOF 0.5960 < XGB 0.6030, 50:50 블렌드로 하락 |
| v15 | 2026-05-23 | XGBoost + CT + 3D 회전 증강 + 5-Fold | OOF 0.6134 | **0.6456** | 10K→50K 증강, v13 대비 +0.0036 ↑ (역대 최고) |
| v16 | 2026-05-24 | v15 + OOF 에러 분석 | OOF 0.6134 | **-** | 모델 변경 없음, 분석용 |
| v17 | 2026-05-24 | XGBoost + 속력 기반 잔차 정규화 | OOF 0.6134 | **0.6478** | 잔차÷(speed×dt), 일반화 개선, v15 대비 +0.0022 ↑ (역대 최고) |
| v18 | 2026-05-25 | 속력 구간별 독립 모델 (Slow/Fast 분리) | OOF 0.6038 | **0.6454** | Fast 4K 데이터 부족으로 일반화 실패, v17 대비 하락 |
| v19 | 2026-05-25 | v17 + 메타 게이팅 | OOF 0.6134 (Gated 0.6360) | **0.6552** | XGBoost 도움 여부 분류기, v17 대비 +0.0074 ↑↑ (역대 최고) |
| v20 | 2026-05-25 | v19 + 다중 CT 앵커 + 각가속도 | - | **0.6542** | dω/dt 노이즈·방향 근사 오류로 v19 대비 -0.0010 ↓ |
| v21 | 2026-05-25 | v20 + 속력 스케일 증강 (s∈[0.5,2.0]) | OOF 0.6258 (Gated 0.6433) | **0.655** | OOF +0.0124 ↑↑ but Dacon ≈ 0 (OOF-Dacon 갭 축소) |
| v22 | 2026-05-25 | v21 + TTA N=8 (회전+스케일) | OOF 0.6258 (Gated 0.6433) | **0.6534** | 속력 정규화로 스케일 TTA 무의미, 게이트 불일치로 하락 |
| v23 | 2026-05-26 | Transformer 시퀀스 모델 + 메타 게이팅 | OOF 0.6225 (Gated 0.6367) | **0.6472** | XGBoost(v19) 대비 OOF +0.009↑ but Dacon -0.008↓, 일반화 미달 |
| v24 | 2026-05-26 | XGBoost + Kalman CA 3-way 블렌드 | OOF 0.6138 (Gated 0.6405) | **0.6536** | CA-KF 앵커 단독 0.3968(악화), cv_smooth 피처는 유효, v19 대비 -0.0016↓ |
| v25 | 2026-05-26 | v19 복귀 + cv_smooth 피처 추가 | OOF 0.6148 (Gated 0.6378) | **0.6544** | CA 앵커 제거·cv_smooth 6개 피처 유지, OOF↑+0.0014 but Dacon↓-0.0008 vs v19 |
| v26 | 2026-05-26 | ω 3프레임 안정화 + speed-trend 블렌드 | OOF 0.6160 (Gated 0.6396) | **0.6536** | 강선회 OOF +0.025↑, but Dacon -0.0008↓. OOF↑-Dacon↓ 패턴 7연속 |
| v27 | 2026-05-26 | Ultra-small GRU (hidden=32, ~5K) + N_AUG=9 | - | **0.634** | hidden=32(5K params) 과소적합, XGB 대비 -0.021↓ |
| v28 | 2026-05-28 | GRU hidden=128 (~68K) + N_AUG=19 (~200K 샘플) | - | **0.6428** | v27 대비 +0.0088↑, but XGB v19(0.6552) 대비 -0.0124↓ |
| v29 | 2026-05-30 | XGBoost 복귀 + 원호 피팅(Circle Fit) 피처 추가 | OOF 0.6166 (Gated 0.6385) | **0.6508** | CF 앵커 단독 R-Hit=0.0105(불량), 피처로도 효과 미미. v19 대비 -0.0044↓ |
| v30 | 2026-05-30 | XGBoost + Pseudo-label 재학습 (2-Phase, 고신뢰 40%) | P1 OOF 0.6138 → P2 OOF 0.6206 (Gated 0.6422) | **0.6554** | Pseudo-label로 OOF +0.0068↑, Dacon v19 대비 +0.0002↑ (역대 최고 타이) |

---

## Changelog

### v30
- `train_group()`: `cf_preds_g/cf_w_g` 파라미터 제거, `extra_X/extra_y` 파라미터 추가, per-fold 테스트 잔차 수집 및 반환
- `main()`: 2-Phase 구조 도입 — Phase1(기본 학습) → 5-fold std 기반 고신뢰 40% 선택 → Phase2(pseudo-label 포함 재학습)
- `make_xgb_features()`: CF 파라미터(`cf_pred`, `cf_weight`) 제거, v25 수준으로 복귀 (383 피처)
- `make_feature_names()`: CF 관련 피처명 제거

### v29
- `predict_circle_fit()`: PCA 투영 + 2D 최소제곱 원 피팅 함수 추가 (n_pts=6)
- `make_xgb_features()`: `cf_pred`, `cf_weight` 파라미터 추가, CF 관련 10개 피처 추가 (cf_pred_L×3, cf_vs_cv_L×3, cf_vs_ct_L×3, cf_weight×1 → 393 피처)
- `make_feature_names()`: CF 피처명 추가
- `train_group()`: `cf_preds_g`, `cf_w_g` 파라미터 추가, 증강 시 CF 예측 회전 적용
- `main()`: GRU 제거, XGBoost 복귀, 원호 피팅 앵커 계산 및 피처 통합

### v28
- `TrajGRU`: hidden 32→128, head `Linear(hidden+7, 64)`→`Linear(hidden+7, 128)` (~68K params)
- `train_gru_5fold()`: N_AUG 9→19 (~200K 증강 샘플)

### v27
- `TrajGRU`: GRU(3→32, 1layer) + head Linear(39,64)+Linear(64,3) (~5K params) 신규 추가
- `train_gru_5fold()`: train_dl_5fold 대체, N_AUG=9, 100 epochs, batch=512
- `main()`: XGBoost 제거, GRU 학습으로 교체 (게이트는 XGBClassifier 유지)

### v26
- `predict_ct()`: ω 단일 프레임 → 마지막 3프레임 평균으로 안정화
- `_speed_trend_inner()`: 접선 가속도 보정 CV 앵커 추가 (trend_w ∈ [0, 0.4])
- `batch_physics_blend()`: speed-trend inner + CT 2-layer 블렌드로 교체
- `make_xgb_features()`: kappa_trend 피처 추가

### v25
- `main()`: CA 앵커 제거 (v24 복귀), cv_smooth 6개 피처 유지
- CT blend 앵커 복원 (v19 수준)

### v24
- `predict_ca_kf()`: Kalman CA 앵커 추가
- `main()`: CV+CT+CA 3-way 블렌드, ca_kf_pred/ca_kf_vs_cv 피처 10개 추가

### v23
- `TrajTransformer`: MultiheadAttention(d=64, h=4) + FFN 신규 추가 (~110K params)
- `train_dl_5fold()`: Transformer 5-fold 학습, N_AUG=10, 150 epochs
- `prepare_dl_inputs()`: vel_seq(10,3) + phys(7) + R(3,3) 입력 준비 함수 추가

### v22
- TTA: N=8 회전+스케일 증강 평균 적용 (결과적으로 스케일 TTA는 속력 정규화로 무의미)

### v21
- 데이터 증강에 속력 스케일 s∈[0.5, 2.0] 추가

### v20
- 다중 CT 앵커(3개) + 각가속도 dω/dt 피처 추가 시도

### v19
- `XGBClassifier` 메타 게이팅 추가: OOF 개선 여부를 분류 후 소프트 게이트 적용
- 블렌드 앵커(CT+CV) 대비 XGBoost 도움 여부 예측

### v17
- 잔차 정규화 타겟: `(true - blend) @ R / disp_scale` (speed×dt 기반 스케일 나누기)

### v15
- `_random_rotation()`: SO(3) 균일 분포 회전 (QR 분해) 추가
- `train_group()`: N_AUG=4 3D 회전 증강 도입, 10K→50K 샘플

### v13
- `train_group()`: 5-Fold KFold OOF 학습 도입, 전체 10K 데이터 활용
- CT 피처를 `make_xgb_features()` 내부로 이동 (ct_pred_L, ct_vs_cv_L, ct_weight)

### v12
- `predict_ct()`: 등속 선회율(CT) 물리 모델 추가, CV+CT 적응 블렌드

### v10
- `make_xgb_features()`: 법선/접선 가속도(a_t, a_n), 다항식 외삽(quad, cubic), 평면 곡률(kappa_xy, kappa_xz) 추가
- `_local_frame_rotation()`: 로컬 좌표계(자기중심) 변환 추가

### v7
- `make_xgb_features()`: 피처 94→274개 확장 (저크, Frenet 곡률, 멀티스케일 CV, 경로 직선성 등)
- `train_group()`: XGBoost 잔차 학습 구조 확립

---

## 메모

### v1
- Train R-Hit 0.9405 vs Public 0.5996 → 과적합 의심
- CV-last train R-Hit 0.5788, Public 미제출

### v2
- 절대 좌표 제거 + 정규화 강화했으나 0.5950으로 v1보다 낮음
- 절대 좌표가 오히려 도움이 됐거나, 복잡도 축소 자체가 성능 하락 원인일 수 있음

### v4
- LSTM (hidden=128, 2layer) + 잔차학습으로 0.6038
- v3 XGBoost(0.609)보다 낮음 → 10,000샘플에서 XGBoost가 LSTM보다 강건

### v3
- 잔차 학습(CV-last 오차 보정)으로 0.609 달성, 현재 최고점
- CV-last 예측 delta를 피처로 추가해 XGBoost가 보정 방향을 앎
- 절대 좌표 피처 복구, 모델 파라미터 v1으로 복구

### v6
- turn_cos(방향 전환 각도), speed_slope, CV-delta 정렬도 등 29개 글로벌 피처
- v5(0.6032) 대비 +0.0012 소폭 상승, v3 XGBoost(0.6090)는 여전히 상회
- BiLSTM FE 강화 한계 확인 → XGBoost 앙상블 또는 XGBoost FE 강화로 전환 검토

### v14
- OOF: XGB 0.6030 / LGB 0.5960 / Ensemble 0.6064
- LightGBM이 XGBoost보다 약함 (OOF -0.007)
- 50:50 블렌드로 XGBoost 성능 희석 → public 0.6406 (v13 대비 -0.0014 ↓)
- 피처 중요도 수치가 이상하게 큼 → XGB(비율 기반)와 LGB(분기 횟수 기반) 스케일이 달라 합산 시 의미 없음
- v13이 여전히 역대 최고

### v13
- 5-Fold CV 도입: 10,000개 전체를 학습에 활용, 5개 모델 잔차 평균으로 테스트 예측
- OOF R-Hit 0.6030 (Fold별: 0.608 / 0.610 / 0.597 / 0.604 / 0.597)
- CT 피처 중요도 2위(ct_pred_z), 3위(ct_pred_y), 4위(ct_vs_cv_z) — v12와 동일하게 유효
- Dacon Public 0.6420 (v12 대비 +0.0086 ↑↑, 역대 최고)

### v12
- CT(등속선회율) 모델: 현재 각속도 ω로 원호 경로 예측, CV와 선회 강도 기반 적응 블렌드
- CT-blend 앵커 자체 R-Hit: 0.5385 (CV-last 0.5788보다 낮음)
- 그러나 CT 피처(ct_pred, ct_vs_cv)가 중요도 2·3·4위 — XGBoost가 선회 방향을 정확히 파악
- CT weight 평균 0.371 → 약 37%의 궤적에서 의미 있는 선회 감지
- val R-Hit 0.6060이지만 public 0.6334 → 일반화 향상

### v11
- ω(각속도), 부호 있는 평면별 곡률(kappa_xy_s/xz_s), 등가속도 예측(ca_pred), 다항식 피팅 RMSE 등 58개 신규 피처 추가 → 364개
- v10(0.6316) 대비 0.003 하락 → FE 포화 확인, 방향 전환 필요

### v7
- XGBoost + 잔차학습으로 복귀, 피처 94개 → 274개로 대폭 확장
- 저크(3차 미분), Frenet 곡률(κ), 회전축, 멀티스케일 CV, 경로 직선성 등 추가
- 0.6240으로 역대 최고 (v3 대비 +0.0150 ↑↑)
- 피처 중요도 → output/feature_importance_v7.csv 저장
