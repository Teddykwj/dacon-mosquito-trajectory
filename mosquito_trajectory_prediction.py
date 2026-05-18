import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
LABELS_CSV = DATA_DIR / "train_labels.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

DT      = 0.04
HORIZON = 2


# ── Logger ─────────────────────────────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mosquito")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "v7_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
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
def _turn_cos(vels: np.ndarray) -> np.ndarray:
    """Cosine similarity between consecutive velocity vectors. (10,)"""
    eps   = 1e-8
    norms = np.linalg.norm(vels, axis=1)
    cos   = np.zeros(len(vels))
    for i in range(1, len(vels)):
        cos[i] = np.dot(vels[i], vels[i - 1]) / (norms[i] * norms[i - 1] + eps)
    return cos


def make_xgb_features(traj: np.ndarray, cv_pred: np.ndarray) -> np.ndarray:
    """274개 피처 벡터"""
    vels      = np.diff(traj, axis=0) / DT                      # (10, 3)
    accs_raw  = np.diff(vels, axis=0) / DT                      # (9, 3)
    accs      = np.vstack([np.zeros((1, 3)), accs_raw])          # (10, 3) zero-pad
    jerk_raw  = np.diff(accs_raw, axis=0) / DT                  # (8, 3)
    jerk      = np.vstack([np.zeros((2, 3)), jerk_raw])          # (10, 3) zero-pad

    speed       = np.linalg.norm(vels, axis=1)                  # (10,)
    acc_mag     = np.linalg.norm(accs, axis=1)                  # (10,)
    jerk_mag    = np.linalg.norm(jerk, axis=1)                  # (10,)
    speed_delta = np.concatenate([[0], np.diff(speed)])          # (10,) zero-pad

    turn_cos = _turn_cos(vels)                                   # (10,)

    # Frenet curvature κ = |v × a| / |v|³  +  rotation axis direction
    kappa     = np.zeros(10)
    rot_axes  = np.zeros((10, 3))
    for i in range(1, 10):
        cross      = np.cross(vels[i], accs[i])
        cross_norm = np.linalg.norm(cross)
        v_norm     = np.linalg.norm(vels[i])
        kappa[i]   = cross_norm / (v_norm ** 3 + 1e-8)
        rot_axes[i] = cross / (cross_norm + 1e-8)

    # Multi-scale CV predictions
    cv_1   = traj[-1] + vels[-1] * DT * HORIZON
    cv_3   = traj[-1] + vels[-3:].mean(0) * DT * HORIZON
    cv_5   = traj[-1] + vels[-5:].mean(0) * DT * HORIZON
    cv_all = traj[-1] + vels.mean(0) * DT * HORIZON
    cv_spread = np.array([np.std([cv_1[i], cv_3[i], cv_5[i]]) for i in range(3)])

    # Geometric summary
    path_len    = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    disp_mag    = np.linalg.norm(traj[-1] - traj[0])
    straightness = disp_mag / (path_len + 1e-8)

    t_idx       = np.arange(10)
    speed_slope = float(np.polyfit(t_idx, speed, 1)[0])

    vel_early    = vels[:5].mean(0)
    vel_late     = vels[5:].mean(0)
    last_vel_unit = vels[-1] / (np.linalg.norm(vels[-1]) + 1e-8)

    cv_delta     = cv_pred - traj[-1]
    cv_align     = np.dot(cv_delta / (np.linalg.norm(cv_delta) + 1e-8), last_vel_unit)

    feats = np.concatenate([
        # ── 시계열 (XGBoost가 스텝별 패턴 학습) ────────────────────────────────
        vels.flatten(),          # (30) 속도 xyz × 10스텝
        accs.flatten(),          # (30) 가속도 xyz × 10스텝
        jerk.flatten(),          # (30) 저크 xyz × 10스텝
        speed,                   # (10) 속력 시계열
        acc_mag,                 # (10) 가속도 크기 시계열
        jerk_mag,                # (10) 저크 크기 시계열
        speed_delta,             # (10) 속력 변화율 시계열
        turn_cos,                # (10) 방향 전환 코사인 시계열
        kappa,                   # (10) 곡률 시계열
        rot_axes.flatten(),      # (30) 회전축 방향 시계열

        # ── 위치 시계열 ─────────────────────────────────────────────────────────
        traj.flatten(),          # (33) 절대 위치 xyz × 11스텝

        # ── 멀티스케일 CV ───────────────────────────────────────────────────────
        cv_pred - traj[-1],      # (3) CV-last delta
        cv_1   - traj[-1],       # (3) 1스텝 CV delta (=cv_pred)
        cv_3   - traj[-1],       # (3) 3스텝 평균 CV delta
        cv_5   - traj[-1],       # (3) 5스텝 평균 CV delta
        cv_all - traj[-1],       # (3) 전체 평균 CV delta
        cv_1   - cv_5,           # (3) 단기 vs 장기 CV 불일치
        cv_spread,               # (3) 스케일 간 분산

        # ── 요약 통계 ───────────────────────────────────────────────────────────
        [turn_cos[1:].mean(), turn_cos[1:].std(),
         turn_cos[1:].min(),  turn_cos[-1]],       # (4) 방향 전환 통계
        [speed.mean(), speed.std(),
         speed[-1],    speed_slope],               # (4) 속력 통계
        [speed_delta[1:].mean(), speed_delta[1:].std()],  # (2) 속력 변화율 통계
        [jerk_mag[2:].mean(), jerk_mag[2:].std(), jerk_mag[2:].max()],  # (3) 저크 통계
        [kappa[1:].mean(), kappa[1:].std(),
         kappa[1:].max(),  kappa[-1]],             # (4) 곡률 통계
        [acc_mag[1:].mean(), acc_mag[1:].std(), acc_mag[1:].max()],  # (3) 가속도 크기 통계
        vel_late - vel_early,    # (3) 전후반 속도 차이
        last_vel_unit,           # (3) 마지막 속도 단위벡터
        [cv_align],              # (1) CV-delta 정렬도
        [straightness],          # (1) 경로 직선성
        traj[-1] - traj[0],      # (3) 전체 변위 벡터
        vels[-3:].mean(0) - vels.mean(0),  # (3) 최근 vs 전체 방향 전환
        accs[1:].mean(0),        # (3) 평균 가속도
        accs[-3:].mean(0),       # (3) 최근 가속도
    ])
    return feats.astype(np.float32)


def make_feature_names() -> list[str]:
    """make_xgb_features 출력 순서와 동일한 피처 이름 리스트"""
    axes = ['x', 'y', 'z']
    names: list[str] = []

    for step in range(10):
        for ax in axes:
            names.append(f"vel_{ax}_t{step}")
    for step in range(10):
        for ax in axes:
            names.append(f"acc_{ax}_t{step}")
    for step in range(10):
        for ax in axes:
            names.append(f"jerk_{ax}_t{step}")
    for step in range(10):
        names.append(f"speed_t{step}")
    for step in range(10):
        names.append(f"acc_mag_t{step}")
    for step in range(10):
        names.append(f"jerk_mag_t{step}")
    for step in range(10):
        names.append(f"speed_delta_t{step}")
    for step in range(10):
        names.append(f"turn_cos_t{step}")
    for step in range(10):
        names.append(f"kappa_t{step}")
    for step in range(10):
        for ax in axes:
            names.append(f"rot_axis_{ax}_t{step}")
    for step in range(11):
        for ax in axes:
            names.append(f"pos_{ax}_t{step}")

    for ax in axes:
        names.append(f"cv_last_delta_{ax}")
    for ax in axes:
        names.append(f"cv_1_delta_{ax}")
    for ax in axes:
        names.append(f"cv_3_delta_{ax}")
    for ax in axes:
        names.append(f"cv_5_delta_{ax}")
    for ax in axes:
        names.append(f"cv_all_delta_{ax}")
    for ax in axes:
        names.append(f"cv_short_long_div_{ax}")
    for ax in axes:
        names.append(f"cv_spread_{ax}")

    names += ["turn_mean", "turn_std", "turn_min", "turn_last"]
    names += ["speed_mean", "speed_std", "speed_last", "speed_slope"]
    names += ["spd_delta_mean", "spd_delta_std"]
    names += ["jerk_mean", "jerk_std", "jerk_max"]
    names += ["kappa_mean", "kappa_std", "kappa_max", "kappa_last"]
    names += ["acc_mag_mean", "acc_mag_std", "acc_mag_max"]
    for ax in axes:
        names.append(f"vel_late_early_{ax}")
    for ax in axes:
        names.append(f"last_vel_unit_{ax}")
    names += ["cv_align", "straightness"]
    for ax in axes:
        names.append(f"disp_{ax}")
    for ax in axes:
        names.append(f"recent_vs_overall_{ax}")
    for ax in axes:
        names.append(f"mean_acc_{ax}")
    for ax in axes:
        names.append(f"recent_acc_{ax}")

    return names


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1) <= 0.01))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 55)
    log.info("모기 비행 궤적 예측 v7 (XGBoost + 피처 확장 + 잔차학습)")
    log.info("=" * 55)

    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)
    log.info(f"Train: {len(train_data)}개  |  Test: {len(test_data)}개")

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    cv_preds_train = batch_cv_last(train_data)
    cv_preds_test  = batch_cv_last(test_data)
    log.info(f"[CV-last]  R-Hit={r_hit(cv_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_preds_train, true_xyz):.2f}cm")

    residuals_train = true_xyz - cv_preds_train

    log.info("피처 생성 중...")
    X_train = np.array([make_xgb_features(t, cv)
                        for t, cv in zip(train_data, cv_preds_train)])
    X_test  = np.array([make_xgb_features(t, cv)
                        for t, cv in zip(test_data, cv_preds_test)])
    feat_names = make_feature_names()
    log.info(f"피처 수: {X_train.shape[1]}")

    # Train / Val 분리
    n   = len(train_data)
    idx = np.random.RandomState(42).permutation(n)
    val_n   = int(n * 0.2)
    val_idx = idx[:val_n]
    tr_idx  = idx[val_n:]

    X_tr,   X_val  = X_train[tr_idx],          X_train[val_idx]
    y_tr,   y_val  = residuals_train[tr_idx],  residuals_train[val_idx]
    val_cv          = cv_preds_train[val_idx]
    val_true        = true_xyz[val_idx]

    base_xgb = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model = MultiOutputRegressor(base_xgb, n_jobs=1)

    log.info("XGBoost 학습 중...")
    model.fit(X_tr, y_tr)

    # Validation
    val_res_pred = model.predict(X_val)
    val_preds    = val_cv + val_res_pred
    log.info(f"[v7-val]   R-Hit={r_hit(val_preds, val_true):.4f}  "
             f"MeanDist={mean_dist_cm(val_preds, val_true):.2f}cm")

    # Train performance
    tr_res_pred = model.predict(X_train)
    tr_preds    = cv_preds_train + tr_res_pred
    log.info(f"[v7-train] R-Hit={r_hit(tr_preds, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(tr_preds, true_xyz):.2f}cm  (과적합 참고용)")

    # Feature importance (평균 across x/y/z)
    importances = np.array(
        [est.feature_importances_ for est in model.estimators_]
    ).mean(axis=0)

    top_n   = 40
    top_idx = np.argsort(importances)[::-1][:top_n]
    log.info(f"\n{'='*55}")
    log.info(f"Top {top_n} 피처 중요도 (x/y/z 평균)")
    log.info(f"{'='*55}")
    for rank, i in enumerate(top_idx, 1):
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        log.info(f"  {rank:2d}. {name:<35s}  {importances[i]:.4f}")

    # 중요도 CSV 저장
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    imp_df = pd.DataFrame({
        'feature': feat_names,
        'importance_x': model.estimators_[0].feature_importances_,
        'importance_y': model.estimators_[1].feature_importances_,
        'importance_z': model.estimators_[2].feature_importances_,
        'importance_mean': importances,
    }).sort_values('importance_mean', ascending=False)
    imp_csv = out_dir / "feature_importance_v7.csv"
    imp_df.to_csv(imp_csv, index=False)
    log.info(f"\n피처 중요도 전체 저장 → {imp_csv}")

    # 제출
    final_test = cv_preds_test + model.predict(X_test)
    sub        = pd.read_csv(SAMPLE_SUB)
    pred_map   = {tid: pred for tid, pred in zip(test_ids, final_test)}
    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )

    out_sub = out_dir / "submission_xgb_v7.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)

    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
