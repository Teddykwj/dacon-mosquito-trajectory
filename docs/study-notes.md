# Study Notes — 모기 궤적 예측 코드 이해

---

## 1. 문제 정의

```
입력: 모기 위치 11개 (t = -400ms ~ 0ms, 40ms 간격)
출력: t = +80ms 의 위치 (x, y, z)
평가: R-Hit@1cm — 예측이 실제 위치 1cm 이내면 정답
```

---

## 2. 핵심 상수

```python
DT      = 0.04  # 프레임 간격 40ms
HORIZON = 2     # 예측 대상까지 2스텝 = 80ms
```

속도 계산: `vel = (pos[t] - pos[t-1]) / DT` → 단위 cm/s

---

## 3. 전체 구조 한눈에 보기

```
관측 궤적 (11개 위치)
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│   CV-last 예측    │     │   XGBoost 예측       │
│  "마지막 속도가   │     │  "궤적 패턴 보고     │
│   유지된다" 가정  │     │   CV-last 오차 보정" │
└────────┬──────────┘     └──────────┬───────────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
             최종 예측 위치
        CV-last 예측 + XGBoost 보정값
```

**v39 최종 구조** (§21 참조)

```
관측 궤적 → Kalman CA → cv_smooth (앵커)
         → 피처 424개 → [MLP ×3seed + Transformer ×3seed + TTA]
                      → 가중 블렌딩 → XGB 게이팅 → 최종 예측
```

---

## 4. CV-last — 베이스라인

```python
vel = (traj[-1] - traj[-2]) / DT   # 마지막 순간 속도
pred = traj[-1] + vel * DT * HORIZON  # 80ms 후 위치
```

**두 가지 역할:**
- 베이스라인 점수 측정 (R-Hit ≈ 0.5788)
- 잔차 학습의 기준점 — XGBoost가 이 예측의 오차를 보정

---

## 5. 잔차 학습 — XGBoost가 하는 일

**핵심 아이디어:** CV-last가 틀린 만큼을 XGBoost가 보정한다.

```
학습 타깃 y = true_xyz - cv_pred   ← "CV-last가 얼마나 틀렸나"
XGBoost 입력 X = 궤적에서 추출한 364개 피처
XGBoost 출력 = 보정값 (Δx, Δy, Δz)

최종 예측 = cv_pred + 보정값
```

**잔차를 타깃으로 쓰는 이유:**
- 절대 위치(10.0cm) 예측보다 오차량(-0.3cm) 예측이 훨씬 쉬움
- CV-last가 이미 좋은 예측을 하므로 XGBoost는 미세 보정만 담당

**CV-last가 주로 틀리는 상황:**

| 상황 | 오류 | 보정 |
|------|------|------|
| 방향 전환 중 | 직진으로 예측 | 꺾인 방향으로 |
| 속도 감소 중 | 너무 멀리 예측 | 가까운 쪽으로 |
| 강한 원심력 | 접선 방향으로 예측 | 법선 방향으로 |
| 등속 직진 | 거의 정확 | 보정 ≈ 0 |

**XGBoost 내부 동작:** 500개의 결정 트리를 순서대로 쌓음. 각 트리는 이전 트리들이 못 줄인 오차를 추가로 줄이는 방향으로 생성 (Gradient Boosting).

---

## 6. 로컬 좌표계 — 방향 정규화

**문제:** 모기 A는 북쪽, 모기 B는 동쪽으로 날면 같은 "직진" 패턴이어도 피처값이 달라짐.

**해결:** 마지막 속도 방향을 항상 +x축으로 정렬하는 회전행렬 R을 만들어 모든 벡터에 적용.

```
회전 전 (글로벌): 북쪽으로 나는 모기의 속도 = (0, 5, 0)
회전 후 (로컬):   앞으로 나는 것으로 통일    = (5, 0, 0)
```

**회전행렬 R (3×3):**
```python
e1 = 마지막속도 방향  → 새 +x축
e2 = e1에 수직       → 새 y축  (외적으로 계산)
e3 = e1, e2에 수직   → 새 z축  (외적으로 계산)
R  = [e1, e2, e3]
```

**변환:**
- 글로벌 → 로컬: `R @ v`
- 로컬 → 글로벌: `R.T @ v` (전치행렬 = 역행렬)

피처도 로컬, 학습 타깃(잔차)도 로컬로 변환 후 학습. 예측 후 R.T로 역변환해서 글로벌 좌표 복원.

---

## 7. 피처 364개 → 424개 — make_xgb_features

궤적 하나 → 숫자 벡터로 변환. (v36에서 383→424개로 확장)

| 그룹 | 피처 | 의미 |
|------|------|------|
| 기본 시계열 | vel, acc, jerk, speed, turn_cos, kappa | 속도·가속도·저크·곡률 시계열 |
| 위치 | traj_L (로컬 상대 위치) | 마지막 위치 기준 11스텝 좌표 |
| 멀티스케일 CV | cv_1/3/5/all/exp | 다양한 스케일로 외삽한 예측 5종 |
| 물리 파생 | a_t, a_n, ω | 접선가속도·법선가속도·각속도 |
| 다항식 외삽 | quad, cubic + RMSE | 2차·3차 곡선 피팅 후 예측 |
| 부호 곡률 | kappa_xy_s, kappa_xz_s | 선회 방향(좌/우, 상/하) |
| CT·CA 물리 앵커 | ct_pred_L, cv_smooth_L 등 | 물리 예측값과 차이 (§15·§16 참조) |
| v36 신규 | torsion, d_omega, 불일치도, 절대위치 | 3D 비틀림, ω 변화율, 앵커 간 불일치 |
| 요약 통계 | mean/std/trend 등 | 각 시계열의 전체 패턴 요약 |

**`make_feature_names`:** 위 피처 각각의 이름 문자열 리스트. 피처 중요도 CSV에 이름으로 저장하기 위해 사용.

---

## 8. 주요 보조 함수

### _turn_cos — 방향 전환 감지
```
cos θ = (v_i · v_{i-1}) / (|v_i| × |v_{i-1}|)

1.0  → 직진
0.0  → 90도 꺾임
-1.0 → U턴
```

### _vel_r2 — 속도의 선형성 측정
```
R² = 1 - (선형피팅 오차) / (전체 분산)

1.0 → 등가속도 운동 (다항식 외삽 신뢰도 높음)
0.0 → 불규칙 운동 (CV-last 보정이 더 중요)
```

### np.linalg — 선형대수 모듈
- `norm(v)` — 벡터 크기 √(x²+y²+z²)
- `norm(vels, axis=1)` — 각 스텝별 속력

---

## 9. CT 모델 (Constant Turn Rate)

**핵심 아이디어:** 모기가 현재 각속도로 원호를 그리며 계속 선회한다고 가정해 예측.

```
ω = a_n / speed   (각속도, rad/s)
               └ 법선가속도 ÷ 속력 = 곡률 반지름의 역수

선회각 θ = ω × 0.08s
ct_weight = clip(θ / (π/4), 0, 1)
         → 약선회(직진) 시 ≈ 0, 강선회 시 → 1

blend = (1 - ct_weight) × cv_pred + ct_weight × ct_pred
```

**핵심 발견:** CT 블렌드 앵커 자체 R-Hit은 0.5385 (CV 0.5788보다 낮음). 하지만 CT 피처(ct_pred, ct_vs_cv)를 XGBoost 입력으로 넣으면 선회 방향 정보를 활용해 최종 예측이 향상됨 → CT 피처 중요도 상위권 진입.

---

## 10. 5-Fold 교차검증 앙상블

**단일 8:2 분할의 한계:** 2,000개 val 데이터가 학습에 참여 못함.

```
5-Fold: 전체 10,000개를 5등분
  → 각 Fold에서 8,000개 학습, 2,000개 검증
  → 5개 모델의 test 예측을 평균
  → OOF(Out-Of-Fold) 점수 = 전체 데이터의 신뢰도 높은 검증
```

v13부터 도입. OOF 0.6030 → Dacon Public 0.6420.

---

## 11. 3D 회전 증강 (v15)

### 핵심 아이디어

모기 궤적 예측은 **방향에 무관해야 한다** — 동일한 비행 패턴이 북쪽으로 날든 동쪽으로 날든 예측 정확도가 같아야 한다. XGBoost는 이 회전 불변성을 스스로 학습하지 못하므로 데이터를 다양한 방향으로 회전시켜 직접 가르친다.

### 랜덤 회전행렬 생성 (QR 분해)
```python
H = rng.randn(3, 3)
Q, R = np.linalg.qr(H)
Q = Q * np.sign(np.diag(R))   # 부호 보정
if det(Q) < 0: Q[:,0] *= -1   # 반사 제거 → SO(3) 보장
```
QR 분해를 쓰는 이유: 3D 공간에 균일하게 분포된 랜덤 회전행렬을 얻기 위해.

### 증강 과정
```
동일한 Q로 궤적 전체를 회전:
  traj_rot  = (Q @ traj.T).T   ← 11개 관측점 전부 회전
  true_rot  = Q @ true_xyz     ← 정답 위치도 동일하게 회전
  blend_rot = Q @ blend        ← 앵커(CV+CT 블렌드)도 회전

ct_weight는 ω = a_n/speed (크기만 사용) → 회전해도 불변 → 재사용

N_AUG = 4 → 10,000 × (1+4) = 50,000개
```

### 누수 방지 Fold 설계
```
val   = 원본 인덱스만  (증강본 포함 금지)
train = 원본(val 제외) + 증강 4벌(val 원본의 증강본도 제외)
```
같은 궤적의 회전본이 train과 val에 동시에 들어가면 데이터 누수 → 원본이 val에 있으면 그 궤적의 모든 증강본도 train에서 제외.

### 효과
| | v13 | v15 |
|--|-----|-----|
| 학습 데이터 | 10,000 | 50,000 |
| OOF R-Hit | 0.6030 | 0.6134 |
| Dacon Public | 0.6420 | 0.6456 |

---

## 12. 물리 블렌드 앵커 (CV + CT 혼합)

XGBoost의 기준점. 두 물리 모델을 선회 강도에 따라 가중 합산한다.

### CV-last (등속 모델)
```python
vel  = (traj[-1] - traj[-2]) / DT      # 마지막 순간 속도
pred = traj[-1] + vel * DT * HORIZON   # 80ms 후 위치 (직선 연장)
```
"지금 속도로 직진하면" 이라는 단순 가정.

### CT (Constant Turn Rate, 등속 선회율 모델)
```python
ω     = a_n / speed        # 법선가속도 ÷ 속력 = 각속도 (rad/s)
theta = ω × 0.08s          # 예측 구간 동안 회전각
# 원호 경로로 위치 계산 (곡선)
ct_weight = clip(theta / (π/4), 0, 1)  # 45도에서 포화
```
"지금 선회율을 유지하면" 이라는 가정. 강한 선회일수록 신뢰.

### 블렌드
```
blend = (1 - ct_weight) × cv_pred + ct_weight × ct_pred
```

| ct_weight | 상황 | blend |
|-----------|------|-------|
| ≈ 0 | 직진 중 | CV-last 그대로 |
| 0.3~0.5 | 약한 선회 | CV + CT 혼합 |
| 1.0 | 강한 선회 | CT 전적 신뢰 |

CT 블렌드 앵커 자체 R-Hit은 0.5385 (CV 0.5788보다 낮음). 하지만 ct_pred, ct_vs_cv를 피처로 제공하면 XGBoost가 선회 방향 정보를 활용해 예측을 크게 개선함.

XGBoost는 `true_xyz - blend` 잔차를 학습하고, 최종 예측은 `blend + xgb_residual`.

---

## 13. 메타 게이팅 (v19)

### 핵심 문제

XGBoost 보정이 모든 샘플에서 도움이 되지 않음.
- 개선 샘플: 6,241개 (62.4%)
- 악화 샘플: 3,759개 (37.6%) → 보정이 오히려 노이즈

### 라벨 생성

OOF 기반이라 누수 없음 (oof_preds는 학습에 참여하지 않은 fold에서 예측된 값).

```python
oof_err   = np.linalg.norm(oof_preds - true_xyz, axis=1)   # XGBoost 최종 오차
blend_err = np.linalg.norm(blend_train - true_xyz, axis=1)  # 물리 앵커 오차

gate_labels = (oof_err < blend_err).astype(int)  # 1=개선, 0=악화
```

샘플마다 "XGBoost 보정 후 오차 < 보정 전 오차"이면 1, 아니면 0.

### 게이트 분류기 학습

```python
gate_clf = XGBClassifier(depth=4, 300 trees, lr=0.05)
gate_clf.fit(X_train, gate_labels)
gate_prob = gate_clf.predict_proba(X_test)[:, 1]  # 0.0 ~ 1.0
```

피처를 보고 "이 샘플에서 XGBoost가 도움이 될 확률"을 출력.

### Soft gate 적용

```python
final = blend + gate_prob × xgb_residual
```

- `gate_prob = 1.0` → XGBoost 보정 100% 적용
- `gate_prob = 0.0` → 블렌드 앵커만 사용
- 중간값 → 부분 적용

### 효과

| | v17 | v19 |
|--|-----|-----|
| OOF (raw) | 0.6134 | 0.6134 |
| OOF (gated, in-sample) | - | 0.6360 |
| Dacon Public | 0.6478 | **0.6552** |

OOF-Gated(0.6360) < Dacon(0.6552) → 테스트에서 더 잘 일반화됨. 게이트가 노이즈성 보정을 차단해 실제 환경에서 효과 극대화.

---

## 14. main() 흐름 (v15 기준)

```
① 데이터 로드
   train_data (10000, 11, 3) + 정답 true_xyz (10000, 3)

② 물리 앵커 계산
   blend = (1-ct_w)*cv_pred + ct_w*ct_pred
   residuals = true_xyz - blend

③ 피처 생성
   X_train (10000, 377)  R_train (10000, 3, 3)

④ 3D 회전 증강 (N_AUG=4)
   X_all (50000, 377)  →  학습 데이터 5배 확장

⑤ 5-Fold CV (누수 방지 설계)
   val = 원본만, train = 원본+증강
   OOF 예측 누적

⑥ 피처 중요도 저장 + 제출 파일 생성
   final_test = blend_test + (5개 모델 예측 평균)
```

---

## 15. Kalman CA 앵커 — cv_smooth (v32~)

**문제:** CT 블렌드(0.537)를 앵커로 쓰면 OOF↑-Dacon↓ 패턴이 7연속 발생.

**해결:** CA(등가속도) 칼만 필터로 11개 관측점을 스무딩해 추정한 속도로 CV 예측.

```python
# _kalman_last_state(): 상태벡터 [pos, vel, acc] forward 필터링
smooth_pos, smooth_vel, smooth_acc = _kalman_last_state(traj)

cv_smooth = smooth_pos + smooth_vel * DT * HORIZON   # 스무딩 속도 기반 CV
```

마지막 2프레임 속도 대신 11프레임 전체를 칼만으로 평활화한 속도를 사용 → 노이즈에 강건.

| 앵커 | R-Hit |
|------|-------|
| CT 블렌드 | 0.537 |
| CV-last | 0.579 |
| **cv_smooth** | **0.581** |

v32에서 앵커를 cv_smooth로 교체하자 OOF와 Dacon이 동시에 상승 (+0.013). **단일 최대 개선.**

---

## 16. 속력 기반 잔차 정규화 (v17~)

**문제:** 빠른 모기(잔차 크기 大)와 느린 모기(잔차 크기 小)의 학습 신호 스케일이 달라 모델이 느린 샘플에 편향됨.

**해결:** 잔차를 `disp_scale = speed_last × DT × HORIZON` 으로 나눠 무차원 비율로 변환.

```python
disp_scale = np.maximum(speed_last * DT * HORIZON, 0.01)   # 최소 1cm

# 학습 타깃 정규화
res_local_norm = (R @ (true - anchor)) / disp_scale

# 예측 후 역정규화
pred_global = R.T @ (pred_norm * disp_scale)
```

회전 증강 시 속력 크기가 불변이므로 `disp_scale`을 증강 샘플에 그대로 재사용할 수 있다.

---

## 17. smooth R-Hit Loss (v37~)

**문제:** MSELoss는 오차 크기를 최소화하지만 평가 지표인 R-Hit@1cm는 임계값 이하 비율을 최대화한다. 두 목표가 불일치.

**해결:** 1cm 임계값 근처에서 미분 가능한 시그모이드 근사.

```python
def smooth_rhit_loss(pred_norm, true_norm, disp_scale, k=10.0):
    diff_cm = norm((pred_norm - true_norm) * disp_scale, dim=1) * 100.0
    return -sigmoid(k * (1.0 - diff_cm)).mean()
```

```
diff_cm = 0cm  → sigmoid(10)  ≈ 1.0  → loss ≈ -1   (최소, 완벽)
diff_cm = 1cm  → sigmoid(0)   = 0.5  → loss = -0.5  (임계)
diff_cm = 2cm  → sigmoid(-10) ≈ 0    → loss ≈ 0     (최대, 실패)
```

k=10이면 1cm 근방 기울기가 충분히 가파르다.

**효과:** MSE 대비 Q5(빠름) R-Hit 0.349→0.464 (+0.115). 임계값에 집중한 손실함수의 효과.

---

## 18. TrajMLP (v37~)

tabular 피처 424개를 입력받아 정규화된 잔차를 예측하는 MLP.

```
Input BN (424)
  → Linear(424, 512) + BN + GELU + Dropout(0.3)
  → Linear(512, 512) + BN + GELU + Dropout(0.3)  ↑ residual 연결
  → Linear(512, 256) + BN + GELU + Dropout(0.3)
  → Linear(256, 3)
```

- **Residual 연결**: 두 번째 레이어 입력+출력을 더함 → 기울기 흐름 안정
- **Input BN**: 속도·곡률 등 단위가 다른 피처를 정규화
- **학습**: AdamW + CosineAnnealingLR, best checkpoint (val R-Hit 기준)
- **Multi-seed ×3**: seed 42/123/456 앙상블로 분산 감소

---

## 19. TrajTransformer + TTA (v39~)

raw velocity sequence(10, 3)를 직접 인코딩하는 Transformer 모델.

### 아키텍처

```
vel_seq (B, 10, 3)
  → Linear(3, 128) + 학습 가능 위치 임베딩(1, 10, 128)
  → TransformerEncoder ×4 레이어
      (d_model=128, nhead=4, FFN=512, Pre-LN, Dropout=0.15)
  → Global Average Pooling  →  (B, 128)
  → concat phys(7)          →  (B, 135)
  → Linear(135, 256) + GELU + Linear(256, 64) + GELU + Linear(64, 3)
```

TrajMLP(tabular 피처)와 상호 보완: MLP는 명시적 물리 피처, Transformer는 raw sequence에서 패턴을 직접 학습.

### TTA (Test Time Augmentation)

추론 시 테스트 샘플마다 32회 랜덤 회전을 적용해 예측한 뒤 원래 좌표계로 역변환해 평균.

```python
# 회전 Q 적용 후 예측 → 역변환
res_rg   = R_Q.T @ (model(Q·traj) * disp_scale)   # Q좌표계 잔차
res_orig = Q.T @ res_rg                             # 원래 좌표계로 복원

# 32회 평균
fold_acc += res_orig   # 누적 후 / n_tta
```

TTA 수학적 근거: 앵커(cv_smooth)는 Q에 무관하므로 `Q.T @ Q @ anchor = anchor`. 잔차 부분만 역변환하면 된다.

**효과:** 단일 방향 예측의 분산 감소. 특히 빠른 속력·급선회처럼 예측이 불안정한 구간에 유효.

---

## 20. Pseudo-label (v30~v34)

**핵심 아이디어:** 테스트 데이터를 가짜 정답(pseudo-label)으로 활용해 모델을 재학습.

```
Phase1: 기본 5-Fold 학습 → 각 Fold별 test 잔차 예측 분산 계산
Phase2: fold간 std 하위 40% (고신뢰) 테스트 샘플을 pseudo-label로 추가 → 재학습
```

**고신뢰 기준:** 5개 fold 예측의 표준편차가 낮을수록 예측이 일관됨 → 정답에 가까울 가능성이 높음.

**효과와 한계:**
- Phase2 OOF: +0.0068 (v30)
- Phase3 이상은 포화 상태 (v34에서 확인)
- v37 MLP+smooth loss 도입 후 pseudo-label 없이도 성능 초과 → 폐기

---

## 21. main() 흐름 (v39 최종)

```
① 데이터 로드
   train_data (10000, 11, 3) + true_xyz (10000, 3)

② 앵커 계산
   cv_smooth = Kalman CA 스무딩 속도 기반 CV  (R-Hit=0.5812)

③ CT 결과 계산 (피처 입력용)
   ct_pred, ct_weight → make_xgb_features() 에서 활용

④ 피처 생성
   X_train (10000, 424)  R_train (10000, 3, 3)

⑤ Multi-seed MLP 앙상블  [seeds=42/123/456]
   각 seed: 5-Fold × N_AUG=9 × smooth R-Hit loss
   → mlp_oof (10000, 3),  mlp_test_res (10000, 3)

⑥ Multi-seed Transformer 앙상블  [seeds=42/123/456]
   각 seed: 5-Fold × N_AUG=9 × smooth R-Hit loss + TTA×32
   → tf_oof (10000, 3),  tf_test_res (10000, 3)

⑦ OOF R-Hit 기반 가중 블렌딩
   w_mlp = mlp_rhit / (mlp_rhit + tf_rhit)
   ens_oof      = w_mlp × mlp_oof  + w_tf × tf_oof
   ens_test_res = w_mlp × mlp_test + w_tf × tf_test

⑧ XGB 메타 게이팅
   gate_labels  = (ens_oof_err < anchor_err)
   gate_prob    = XGBClassifier.predict_proba(X_test)[:, 1]

⑨ 최종 예측
   final = cv_smooth_test + gate_prob × ens_test_res
   → NaN/Inf → anchor 폴백
   → ±10cm 클리핑
```
