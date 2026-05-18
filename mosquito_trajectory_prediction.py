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
    fh = logging.FileHandler(log_dir / "v9_log.txt", encoding="utf-8")
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


# ── Feature engineering helpers ────────────────────────────────────────────────
def _local_frame_rotation(vel: np.ndarray) -> np.ndarray:
    """
    마지막 속도벡터를 +x 방향으로 정렬하는 회전행렬 R (3×3).
    x_local = R @ x_global  /  x_global = R.T @ x_local
    """
    eps = 1e-8
    e1  = vel / (np.linalg.norm(vel) + eps)          # forward (마지막 속도 방향)
    ref = np.array([0., 0., 1.])                      # z-up 기준
    if abs(e1[2]) > 0.9:                              # e1 ≈ z 이면 기준 교체
        ref = np.array([0., 1., 0.])
    e2  = np.cross(e1, ref)
    e2 /= np.linalg.norm(e2) + eps
    e3  = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=0)             # (3, 3)


def _turn_cos(vels: np.ndarray) -> np.ndarray:
    """연속 속도벡터 간 코사인 유사도. 첫 스텝은 0."""
    eps   = 1e-8
    norms = np.linalg.norm(vels, axis=1)
    cos   = np.zeros(len(vels))
    for i in range(1, len(vels)):
        cos[i] = np.dot(vels[i], vels[i - 1]) / (norms[i] * norms[i - 1] + eps)
    return cos


def make_xgb_features(traj: np.ndarray,
                       cv_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    로컬 프레임(마지막 속도 → +x) 기준 피처 238개와 회전행렬 R을 반환.
    R: global → local  (R.T: local → global)
    """
    vels_g = np.diff(traj, axis=0) / DT              # (10, 3) 글로벌
    R      = _local_frame_rotation(vels_g[-1])        # (3, 3)

    # ── 로컬 프레임으로 변환 ──────────────────────────────────────────────────
    vels     = vels_g @ R.T                           # (10, 3)
    accs_raw = np.diff(vels, axis=0) / DT             # (9, 3)
    accs     = np.vstack([np.zeros((1, 3)), accs_raw])  # (10, 3)
    jerk_raw = np.diff(accs_raw, axis=0) / DT         # (8, 3)
    jerk     = np.vstack([np.zeros((2, 3)), jerk_raw])  # (10, 3)
    traj_L   = (traj - traj[-1]) @ R.T               # (11, 3) 마지막 점 중심

    # ── 스칼라(회전 불변) ────────────────────────────────────────────────────
    speed       = np.linalg.norm(vels, axis=1)        # (10,)
    acc_mag     = np.linalg.norm(accs, axis=1)        # (10,)
    jerk_mag    = np.linalg.norm(jerk, axis=1)        # (10,)
    speed_delta = np.diff(speed)                      # (9,)
    turn_cos    = _turn_cos(vels)                     # (10,)

    kappa = np.zeros(10)
    for i in range(1, 10):
        cross    = np.cross(vels[i], accs[i])
        v_norm   = np.linalg.norm(vels[i])
        kappa[i] = np.linalg.norm(cross) / (v_norm ** 3 + 1e-8)

    # ── 멀티스케일 CV (로컬 프레임 변위) ────────────────────────────────────
    cv_L     = vels[-1] * DT * HORIZON               # (3) last-vel extrapolation
    cv_3_L   = vels[-3:].mean(0) * DT * HORIZON
    cv_5_L   = vels[-5:].mean(0) * DT * HORIZON
    cv_all_L = vels.mean(0) * DT * HORIZON
    cv_spread = np.array([np.std([cv_L[i], cv_3_L[i], cv_5_L[i]]) for i in range(3)])

    w      = np.array([(0.7) ** i for i in range(9, -1, -1)])
    w     /= w.sum()
    cv_exp_L = (vels * w[:, None]).sum(0) * DT * HORIZON

    # ── 요약 피처 ───────────────────────────────────────────────────────────
    path_len     = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    disp_mag     = np.linalg.norm(traj[-1] - traj[0])
    straightness = disp_mag / (path_len + 1e-8)

    t_idx        = np.arange(10)
    speed_slope  = float(np.polyfit(t_idx, speed, 1)[0])

    vel_early          = vels[:5].mean(0)
    vel_late           = vels[5:].mean(0)
    vel_last2_vs_last5 = vels[-2:].mean(0) - vels[-5:].mean(0)
    vel_last3_vs_last7 = vels[-3:].mean(0) - vels[-7:].mean(0)

    # 로컬 프레임에서 cv_3 방향과 last-vel 방향의 정렬도 (직진 vs 선회 판별)
    last_vel_unit = vels[-1] / (np.linalg.norm(vels[-1]) + 1e-8)  # ≈ [1,0,0]
    cv_3_unit     = cv_3_L   / (np.linalg.norm(cv_3_L)   + 1e-8)
    cv_align      = float(np.dot(cv_3_unit, last_vel_unit))        # 3-step CV vs 1-step 정렬도

    disp_L      = traj_L[-1] - traj_L[0]                          # (3) 로컬 변위벡터
    turn_trend  = turn_cos[-3:].mean() - turn_cos[1:7].mean()
    kappa_trend = kappa[-3:].mean() - kappa[1:7].mean()
    speed_ratio = speed[-1] / (speed.mean() + 1e-8)

    feats = np.concatenate([
        # ── 시계열 (로컬 프레임) ─────────────────────────────────────────────
        vels.flatten(),          # (30)
        accs[1:].flatten(),      # (27)
        jerk[2:].flatten(),      # (24)
        speed,                   # (10)
        acc_mag[1:],             # (9)
        jerk_mag[2:],            # (8)
        speed_delta,             # (9)
        turn_cos[1:],            # (9)
        kappa[1:],               # (9)

        # ── 위치 (로컬 프레임, 마지막 점 중심) ──────────────────────────────
        traj_L.flatten(),        # (33)

        # ── 멀티스케일 CV (로컬 변위) ────────────────────────────────────────
        cv_L,                    # (3)
        cv_3_L,                  # (3)
        cv_5_L,                  # (3)
        cv_all_L,                # (3)
        cv_exp_L,                # (3)
        cv_L - cv_5_L,           # (3) 단기 vs 장기 불일치
        cv_spread,               # (3)

        # ── 요약 통계 ────────────────────────────────────────────────────────
        [turn_cos[1:].mean(), turn_cos[1:].std(),
         turn_cos[1:].min(),  turn_cos[-1]],
        [speed.mean(), speed.std(), speed[-1], speed_slope],
        [speed_delta.mean(), speed_delta.std()],
        [jerk_mag[2:].mean(), jerk_mag[2:].std(), jerk_mag[2:].max()],
        [kappa[1:].mean(), kappa[1:].std(), kappa[1:].max(), kappa[-1]],
        [acc_mag[1:].mean(), acc_mag[1:].std(), acc_mag[1:].max()],
        vel_late - vel_early,      # (3)
        vel_last2_vs_last5,        # (3)
        vel_last3_vs_last7,        # (3)
        last_vel_unit,             # (3) ≈ [1,0,0] 상수에 가깝지만 유지
        [cv_align],                # (1) 3-step vs 1-step CV 정렬도
        [straightness],            # (1)
        disp_L,                    # (3) 로컬 변위벡터
        vels[-3:].mean(0) - vels.mean(0),  # (3)
        accs[1:].mean(0),          # (3)
        accs[-3:].mean(0),         # (3)
        [turn_trend],              # (1)
        [kappa_trend],             # (1)
        [speed_ratio],             # (1)
    ])
    return feats.astype(np.float32), R


def make_feature_names() -> list[str]:
    axes  = ['x', 'y', 'z']
    names: list[str] = []

    for step in range(10):
        for ax in axes:
            names.append(f"vel_{ax}_t{step}")
    for step in range(1, 10):
        for ax in axes:
            names.append(f"acc_{ax}_t{step}")
    for step in range(2, 10):
        for ax in axes:
            names.append(f"jerk_{ax}_t{step}")
    for step in range(10):
        names.append(f"speed_t{step}")
    for step in range(1, 10):
        names.append(f"acc_mag_t{step}")
    for step in range(2, 10):
        names.append(f"jerk_mag_t{step}")
    for step in range(9):
        names.append(f"speed_delta_t{step + 1}")
    for step in range(1, 10):
        names.append(f"turn_cos_t{step}")
    for step in range(1, 10):
        names.append(f"kappa_t{step}")
    for step in range(11):
        for ax in axes:
            names.append(f"pos_{ax}_t{step}")

    for ax in axes:
        names.append(f"cv_last_delta_{ax}")
    for ax in axes:
        names.append(f"cv_3_delta_{ax}")
    for ax in axes:
        names.append(f"cv_5_delta_{ax}")
    for ax in axes:
        names.append(f"cv_all_delta_{ax}")
    for ax in axes:
        names.append(f"cv_exp_delta_{ax}")
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
        names.append(f"vel_last2_vs_last5_{ax}")
    for ax in axes:
        names.append(f"vel_last3_vs_last7_{ax}")
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
    names += ["turn_trend", "kappa_trend", "speed_ratio"]

    return names


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1) <= 0.01))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 64)
    log.info("모기 비행 궤적 예측 v9 (XGBoost + 로컬 좌표계 + 잔차학습)")
    log.info("=" * 64)

    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)
    log.info(f"Train: {len(train_data)}개  |  Test: {len(test_data)}개")

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    cv_preds_train = batch_cv_last(train_data)
    cv_preds_test  = batch_cv_last(test_data)
    log.info(f"[CV-last]  R-Hit={r_hit(cv_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_preds_train, true_xyz):.2f}cm")

    residuals_train = true_xyz - cv_preds_train           # (N, 3) 글로벌

    log.info("피처 생성 중 (로컬 프레임)...")
    train_out  = [make_xgb_features(t, cv)
                  for t, cv in zip(train_data, cv_preds_train)]
    X_train    = np.array([o[0] for o in train_out])      # (N, 238)
    R_train    = np.array([o[1] for o in train_out])      # (N, 3, 3)

    test_out   = [make_xgb_features(t, cv)
                  for t, cv in zip(test_data, cv_preds_test)]
    X_test     = np.array([o[0] for o in test_out])
    R_test     = np.array([o[1] for o in test_out])

    feat_names = make_feature_names()
    log.info(f"피처 수: {X_train.shape[1]}")
    assert X_train.shape[1] == len(feat_names), \
        f"피처 수 불일치: {X_train.shape[1]} vs {len(feat_names)}"

    # 잔차를 로컬 프레임으로 변환: res_local[n] = R[n] @ res_global[n]
    residuals_local = np.einsum('nij,nj->ni', R_train, residuals_train)

    # Train / Val 분리
    n       = len(train_data)
    idx     = np.random.RandomState(42).permutation(n)
    val_n   = int(n * 0.2)
    val_idx = idx[:val_n]
    tr_idx  = idx[val_n:]

    X_tr,  X_val  = X_train[tr_idx],        X_train[val_idx]
    y_tr,  y_val  = residuals_local[tr_idx], residuals_local[val_idx]
    R_val         = R_train[val_idx]
    val_cv        = cv_preds_train[val_idx]
    val_true      = true_xyz[val_idx]

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

    # Val 평가: 로컬 → 글로벌 역변환
    val_res_local = model.predict(X_val)                   # (N, 3) 로컬
    val_res_global = np.einsum('nji,nj->ni', R_val, val_res_local)  # R.T @ res
    val_preds     = val_cv + val_res_global
    log.info(f"[v9-val]   R-Hit={r_hit(val_preds, val_true):.4f}  "
             f"MeanDist={mean_dist_cm(val_preds, val_true):.2f}cm")

    # Train 전체 성능
    tr_res_local  = model.predict(X_train)
    tr_res_global = np.einsum('nji,nj->ni', R_train, tr_res_local)
    tr_preds      = cv_preds_train + tr_res_global
    log.info(f"[v9-train] R-Hit={r_hit(tr_preds, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(tr_preds, true_xyz):.2f}cm  (과적합 참고용)")

    # Feature importance
    importances = np.array(
        [est.feature_importances_ for est in model.estimators_]
    ).mean(axis=0)
    top_n   = 40
    top_idx = np.argsort(importances)[::-1][:top_n]
    log.info(f"\n{'='*64}")
    log.info(f"Top {top_n} 피처 중요도 (x/y/z 평균)")
    log.info(f"{'='*64}")
    for rank, i in enumerate(top_idx, 1):
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        log.info(f"  {rank:2d}. {name:<38s}  {importances[i]:.4f}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    imp_df = pd.DataFrame({
        'feature':         feat_names,
        'importance_x':    model.estimators_[0].feature_importances_,
        'importance_y':    model.estimators_[1].feature_importances_,
        'importance_z':    model.estimators_[2].feature_importances_,
        'importance_mean': importances,
    }).sort_values('importance_mean', ascending=False)
    imp_csv = out_dir / "feature_importance_v9.csv"
    imp_df.to_csv(imp_csv, index=False)
    log.info(f"\n피처 중요도 전체 저장 → {imp_csv}")

    # 제출: 로컬 잔차 예측 → 글로벌 역변환
    test_res_local  = model.predict(X_test)
    test_res_global = np.einsum('nji,nj->ni', R_test, test_res_local)
    final_test      = cv_preds_test + test_res_global

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_test)}
    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )

    out_sub = out_dir / "submission_xgb_v9.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)

    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
