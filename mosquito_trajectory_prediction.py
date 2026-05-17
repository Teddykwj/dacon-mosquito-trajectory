import logging
import os
import sys
from datetime import datetime
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


# ── Logger ─────────────────────────────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mosquito")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ── Data loading ───────────────────────────────────────────────────────────────
def load_sample(path: Path) -> np.ndarray:
    return pd.read_csv(path)[['x', 'y', 'z']].values  # (11, 3)


def load_dir(directory: Path) -> tuple[list[str], list[np.ndarray]]:
    paths = sorted(Path(directory).glob("*.csv"))
    ids   = [p.stem for p in paths]
    data  = [load_sample(p) for p in paths]
    return ids, data


# ── CV-last ────────────────────────────────────────────────────────────────────
def predict_cv_last(traj: np.ndarray) -> np.ndarray:
    vel = (traj[-1] - traj[-2]) / DT
    return traj[-1] + vel * DT * HORIZON


def batch_cv_last(data: list[np.ndarray]) -> np.ndarray:
    return np.array([predict_cv_last(t) for t in data])


# ── Feature engineering ────────────────────────────────────────────────────────
def extract_features(traj: np.ndarray, cv_pred: np.ndarray) -> np.ndarray:
    """
    v3: CV-last 예측값을 피처로 추가
    → XGBoost가 CV-last의 보정량(잔차)을 학습하도록 앵커 역할
    """
    vels  = np.diff(traj, axis=0) / DT   # (10, 3)
    accs  = np.diff(vels, axis=0) / DT   # (9, 3)

    feats = []

    # 절대 위치 (v1에서 유효했음)
    feats.append(traj[-1])                             # (3,)

    # CV-last 예측 delta (잔차 학습의 앵커)
    feats.append(cv_pred - traj[-1])                   # (3,)

    # 속도 시계열 전체 (10 x 3 = 30)
    feats.append(vels.flatten())

    # 가속도 시계열 전체 (9 x 3 = 27)
    feats.append(accs.flatten())

    # 속도 통계
    feats.append(vels.mean(axis=0))
    feats.append(vels.std(axis=0))

    # 최근 속도 vs 전체 평균
    recent_vel = vels[-3:].mean(axis=0)
    feats.append(recent_vel)
    feats.append(recent_vel - vels.mean(axis=0))

    # 가속도 통계
    feats.append(accs.mean(axis=0))
    feats.append(accs[-3:].mean(axis=0))

    # 전체 변위
    feats.append(traj[-1] - traj[0])

    # 속력
    speeds = np.linalg.norm(vels, axis=1)
    feats.append(speeds)
    feats.append(np.array([speeds.mean(), speeds.std(), speeds[-1]]))

    return np.concatenate(feats)


def build_features(data: list[np.ndarray], cv_preds: np.ndarray) -> np.ndarray:
    return np.array([extract_features(t, cv) for t, cv in zip(data, cv_preds)])


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray, threshold: float = 0.01) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1) <= threshold))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 50)
    log.info("모기 비행 궤적 예측 v3 (잔차 학습)")
    log.info("=" * 50)

    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)
    log.info(f"Train 샘플: {len(train_data)}개  |  Test 샘플: {len(test_data)}개")

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    # ── CV-last 기준선 ─────────────────────────────────────────────────────────
    cv_preds_train = batch_cv_last(train_data)
    cv_preds_test  = batch_cv_last(test_data)
    log.info(f"[CV-last]  R-Hit={r_hit(cv_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_preds_train, true_xyz):.2f}cm")

    # ── 피처 / 타깃 구성 ───────────────────────────────────────────────────────
    X_train = build_features(train_data, cv_preds_train)
    X_test  = build_features(test_data,  cv_preds_test)
    log.info(f"Feature dim: {X_train.shape[1]}")

    # 타깃: CV-last 대비 잔차 (delta → residual)
    residual_train = true_xyz - cv_preds_train
    log.info(f"Residual 크기: mean={np.linalg.norm(residual_train, axis=1).mean()*100:.2f}cm  "
             f"std={np.linalg.norm(residual_train, axis=1).std()*100:.2f}cm")

    # ── XGBoost 학습 ───────────────────────────────────────────────────────────
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
    log.info(f"XGBoost 학습 시작  params={xgb_params}")

    model = MultiOutputRegressor(XGBRegressor(**xgb_params), n_jobs=3)
    model.fit(X_train, residual_train)
    log.info("XGBoost 학습 완료")

    # 최종 예측 = CV-last + 잔차 보정
    residual_pred_train = model.predict(X_train)
    final_preds_train   = cv_preds_train + residual_pred_train
    log.info(f"[XGB-v3-train] R-Hit={r_hit(final_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(final_preds_train, true_xyz):.2f}cm  (과적합 참고용)")

    # ── 오차 상위 10개 ─────────────────────────────────────────────────────────
    dists = np.linalg.norm(final_preds_train - true_xyz, axis=1) * 100
    log.info("-" * 50)
    log.info("오차 상위 10개:")
    for i in np.argsort(dists)[::-1][:10]:
        log.info(f"  {train_ids[i]}  CV={np.linalg.norm(cv_preds_train[i]-true_xyz[i])*100:.2f}cm  "
                 f"XGB={dists[i]:.2f}cm")

    # ── 제출 파일 ──────────────────────────────────────────────────────────────
    residual_pred_test = model.predict(X_test)
    final_preds_test   = cv_preds_test + residual_pred_test

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_preds_test)}

    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_sub = out_dir / "submission_xgb_v3.csv"
    sub.to_csv(out_sub, index=False)

    # 컨테이너가 root로 실행되므로 호스트에서 삭제 가능하도록 권한 개방
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)

    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
