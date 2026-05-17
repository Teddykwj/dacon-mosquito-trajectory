import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
LABELS_CSV = DATA_DIR / "train_labels.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

DT      = 0.04   # 40 ms
HORIZON = 2      # 2 steps → +80 ms


# ── Data loading ───────────────────────────────────────────────────────────────
def load_sample(path: Path) -> np.ndarray:
    return pd.read_csv(path)[['x', 'y', 'z']].values  # (11, 3)


def load_dir(directory: Path) -> tuple[list[str], list[np.ndarray]]:
    paths = sorted(Path(directory).glob("*.csv"))
    ids   = [p.stem for p in paths]
    data  = [load_sample(p) for p in paths]
    return ids, data


# ── Feature engineering ────────────────────────────────────────────────────────
def extract_features(traj: np.ndarray) -> np.ndarray:
    """
    traj: (11, 3)
    반환: 1D feature vector

    설계 원칙: 절대 좌표 대신 상대 변위 위주로 구성
    → train/test 환경이 달라도 운동 패턴은 유지됨
    """
    vels  = np.diff(traj, axis=0) / DT          # (10, 3) - 속도
    accs  = np.diff(vels, axis=0) / DT           # (9, 3)  - 가속도

    feats = []

    # 마지막 관측 위치 (t=0)
    feats.append(traj[-1])                        # (3,)

    # 전체 속도 시계열 (10 x 3 = 30)
    feats.append(vels.flatten())

    # 전체 가속도 시계열 (9 x 3 = 27)
    feats.append(accs.flatten())

    # 속도 통계
    feats.append(vels.mean(axis=0))              # 평균 속도 (3,)
    feats.append(vels.std(axis=0))               # 속도 표준편차 (3,)

    # 최근 속도 vs 전체 평균 (방향 전환 감지)
    recent_vel = vels[-3:].mean(axis=0)
    feats.append(recent_vel)                      # (3,)
    feats.append(recent_vel - vels.mean(axis=0)) # 최근 추세 변화 (3,)

    # 가속도 통계
    feats.append(accs.mean(axis=0))              # (3,)
    feats.append(accs[-3:].mean(axis=0))         # 최근 가속도 (3,)

    # 이동 궤적 전체 변위 (t=-400 → t=0)
    feats.append(traj[-1] - traj[0])             # (3,)

    # 속도 크기 (방향 무관 속력)
    speeds = np.linalg.norm(vels, axis=1)        # (10,)
    feats.append(speeds)
    feats.append(np.array([speeds.mean(), speeds.std(), speeds[-1]]))

    return np.concatenate(feats)


def build_features(data: list[np.ndarray]) -> np.ndarray:
    return np.array([extract_features(t) for t in data])


# ── 기준선: CV-last ────────────────────────────────────────────────────────────
def predict_cv_last(traj: np.ndarray) -> np.ndarray:
    vel = (traj[-1] - traj[-2]) / DT
    return traj[-1] + vel * DT * HORIZON


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray, threshold: float = 0.01) -> float:
    dists = np.linalg.norm(preds - trues, axis=1)
    return float(np.mean(dists <= threshold))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    # 타깃: 절대 좌표가 아니라 t=0 대비 변화량 (delta)
    # → 환경이 달라도 운동 패턴만 학습
    last_pos_train = np.array([t[-1] for t in train_data])
    delta_train    = true_xyz - last_pos_train          # (N, 3)

    X_train = build_features(train_data)
    X_test  = build_features(test_data)

    print(f"Train: {X_train.shape},  Test: {X_test.shape}")
    print(f"Feature dim: {X_train.shape[1]}")

    # ── 기준선 성능 ────────────────────────────────────────────────────────────
    cv_preds = np.array([predict_cv_last(t) for t in train_data])
    print(f"\nCV-last  R-Hit={r_hit(cv_preds, true_xyz):.4f}  "
          f"MeanDist={mean_dist_cm(cv_preds, true_xyz):.2f}cm")

    # ── XGBoost 학습 ───────────────────────────────────────────────────────────
    # 각 축(x, y, z)을 독립적으로 예측
    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(XGBRegressor(**xgb_params), n_jobs=3)
    model.fit(X_train, delta_train)

    # 학습 데이터 성능 (과적합 확인용)
    delta_pred_train = model.predict(X_train)
    xgb_preds_train  = last_pos_train + delta_pred_train
    print(f"XGB-train  R-Hit={r_hit(xgb_preds_train, true_xyz):.4f}  "
          f"MeanDist={mean_dist_cm(xgb_preds_train, true_xyz):.2f}cm")

    # ── 샘플별 상세 ────────────────────────────────────────────────────────────
    print(f"\n{'Sample':<14} {'CV-last':>10} {'XGB':>10}")
    print("-" * 36)
    for i, tid in enumerate(train_ids):
        d_cv  = np.linalg.norm(cv_preds[i] - true_xyz[i]) * 100
        d_xgb = np.linalg.norm(xgb_preds_train[i] - true_xyz[i]) * 100
        m_cv  = "✓" if d_cv  <= 1.0 else "✗"
        m_xgb = "✓" if d_xgb <= 1.0 else "✗"
        print(f"{tid:<14} {m_cv}{d_cv:>7.2f}cm  {m_xgb}{d_xgb:>7.2f}cm")

    # ── 테스트 예측 및 제출 ────────────────────────────────────────────────────
    delta_pred_test = model.predict(X_test)
    last_pos_test   = np.array([t[-1] for t in test_data])
    xgb_preds_test  = last_pos_test + delta_pred_test

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, xgb_preds_test)}

    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )

    out_sub = Path("submission_xgb.csv")
    sub.to_csv(out_sub, index=False)
    print(f"\n제출 파일 → {out_sub}")


if __name__ == '__main__':
    main()
