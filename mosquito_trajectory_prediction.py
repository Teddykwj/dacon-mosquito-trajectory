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
    fh = logging.FileHandler(log_dir / "v10_log.txt", encoding="utf-8")
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
    """마지막 속도를 +x로 정렬하는 회전행렬 R(3×3). global→local."""
    eps = 1e-8
    e1  = vel / (np.linalg.norm(vel) + eps)
    ref = np.array([0., 0., 1.])
    if abs(e1[2]) > 0.9:
        ref = np.array([0., 1., 0.])
    e2  = np.cross(e1, ref);  e2 /= np.linalg.norm(e2) + eps
    e3  = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=0)


def _turn_cos(vels: np.ndarray) -> np.ndarray:
    eps = 1e-8
    norms = np.linalg.norm(vels, axis=1)
    cos   = np.zeros(len(vels))
    for i in range(1, len(vels)):
        cos[i] = np.dot(vels[i], vels[i-1]) / (norms[i]*norms[i-1] + eps)
    return cos


def _vel_r2(vels: np.ndarray) -> np.ndarray:
    """각 축 속도 시계열에 대한 선형 적합도 R² (3,)."""
    t   = np.arange(len(vels), dtype=float)
    t_c = t - t.mean()
    r2  = np.zeros(3)
    for k in range(3):
        y   = vels[:, k]
        y_c = y - y.mean()
        ss_tot = np.dot(y_c, y_c) + 1e-10
        b      = np.dot(t_c, y_c) / (np.dot(t_c, t_c) + 1e-10)
        ss_res = np.sum((y_c - b * t_c) ** 2)
        r2[k]  = max(0.0, 1.0 - ss_res / ss_tot)
    return r2


def make_xgb_features(traj: np.ndarray,
                       cv_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """로컬 프레임 기준 피처 305개 + 회전행렬 R 반환."""
    vels_g = np.diff(traj, axis=0) / DT              # (10, 3) 글로벌
    R      = _local_frame_rotation(vels_g[-1])

    # ── 로컬 프레임 변환 ─────────────────────────────────────────────────────
    vels     = vels_g @ R.T                           # (10, 3)
    accs_raw = np.diff(vels, axis=0) / DT             # (9, 3)
    accs     = np.vstack([np.zeros((1, 3)), accs_raw])  # (10, 3)
    jerk_raw = np.diff(accs_raw, axis=0) / DT         # (8, 3)
    jerk     = np.vstack([np.zeros((2, 3)), jerk_raw])
    traj_L   = (traj - traj[-1]) @ R.T               # (11, 3) 마지막 점 중심

    # ── 스칼라 (회전 불변) ───────────────────────────────────────────────────
    speed       = np.linalg.norm(vels, axis=1)
    acc_mag     = np.linalg.norm(accs, axis=1)
    jerk_mag    = np.linalg.norm(jerk, axis=1)
    speed_delta = np.diff(speed)
    turn_cos    = _turn_cos(vels)

    kappa = np.zeros(10)
    for i in range(1, 10):
        cross    = np.cross(vels[i], accs[i])
        v_norm   = np.linalg.norm(vels[i])
        kappa[i] = np.linalg.norm(cross) / (v_norm**3 + 1e-8)

    # ── [NEW 1] 법선/접선 가속도 분리 ───────────────────────────────────────
    a_t = np.zeros(9)   # 접선: 속도 방향 가속도 (가감속)
    a_n = np.zeros(9)   # 법선: 수직 가속도 (선회력)
    for i in range(1, 10):
        v_unit = vels[i] / (np.linalg.norm(vels[i]) + 1e-8)
        at     = np.dot(accs[i], v_unit)
        an     = np.sqrt(max(0.0, np.dot(accs[i], accs[i]) - at**2))
        a_t[i-1] = at
        a_n[i-1] = an

    # ── [NEW 2] 다항식 외삽 예측 ────────────────────────────────────────────
    t_steps = np.arange(11) * DT
    t_pred  = t_steps[-1] + HORIZON * DT
    quad_L  = np.zeros(3)
    cubic_L = np.zeros(3)
    for k in range(3):
        q2 = np.polyfit(t_steps, traj_L[:, k], 2)
        q3 = np.polyfit(t_steps, traj_L[:, k], 3)
        quad_L[k]  = np.polyval(q2, t_pred)
        cubic_L[k] = np.polyval(q3, t_pred)
    cv_L = vels[-1] * DT * HORIZON                   # CV-last 로컬 변위

    # ── [NEW 3] 평면별 곡률 (로컬 프레임) ───────────────────────────────────
    kappa_xy = np.zeros(9)   # 수평 선회 (yaw)
    kappa_xz = np.zeros(9)   # 수직 선회 (pitch)
    for i in range(1, 10):
        v, a = vels[i], accs[i]
        # xy-plane: 2D cross product
        c_xy = v[0]*a[1] - v[1]*a[0]
        v_xy = np.sqrt(v[0]**2 + v[1]**2)
        kappa_xy[i-1] = abs(c_xy) / (v_xy**3 + 1e-8)
        # xz-plane
        c_xz = v[0]*a[2] - v[2]*a[0]
        v_xz = np.sqrt(v[0]**2 + v[2]**2)
        kappa_xz[i-1] = abs(c_xz) / (v_xz**3 + 1e-8)

    # ── [NEW 4] 속도 선형 적합도 R² ─────────────────────────────────────────
    vel_r2 = _vel_r2(vels)                            # (3,)

    # ── [NEW 5] 가속도 선형 추세 (per axis) ─────────────────────────────────
    t9 = np.arange(9, dtype=float)
    acc_slope = np.array([np.polyfit(t9, accs_raw[:, k], 1)[0] for k in range(3)])

    # ── 멀티스케일 CV (로컬 변위) ────────────────────────────────────────────
    cv_3_L   = vels[-3:].mean(0) * DT * HORIZON
    cv_5_L   = vels[-5:].mean(0) * DT * HORIZON
    cv_all_L = vels.mean(0) * DT * HORIZON
    cv_spread = np.array([np.std([cv_L[i], cv_3_L[i], cv_5_L[i]]) for i in range(3)])
    w         = np.array([(0.7)**i for i in range(9, -1, -1)]); w /= w.sum()
    cv_exp_L  = (vels * w[:, None]).sum(0) * DT * HORIZON

    # ── 기하 요약 ────────────────────────────────────────────────────────────
    path_len     = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    straightness = np.linalg.norm(traj[-1]-traj[0]) / (path_len + 1e-8)
    speed_slope  = float(np.polyfit(np.arange(10), speed, 1)[0])
    vel_early    = vels[:5].mean(0);  vel_late = vels[5:].mean(0)
    vel_l2l5     = vels[-2:].mean(0) - vels[-5:].mean(0)
    vel_l3l7     = vels[-3:].mean(0) - vels[-7:].mean(0)
    last_vel_unit = vels[-1] / (np.linalg.norm(vels[-1]) + 1e-8)
    cv_3_unit     = cv_3_L / (np.linalg.norm(cv_3_L) + 1e-8)
    cv_align      = float(np.dot(cv_3_unit, last_vel_unit))
    disp_L        = traj_L[-1] - traj_L[0]
    turn_trend    = turn_cos[-3:].mean() - turn_cos[1:7].mean()
    kappa_trend   = kappa[-3:].mean()   - kappa[1:7].mean()
    speed_ratio   = speed[-1] / (speed.mean() + 1e-8)

    feats = np.concatenate([
        # ── 기존 시계열 ──────────────────────────────────────────────────────
        vels.flatten(),           # (30)
        accs[1:].flatten(),       # (27)
        jerk[2:].flatten(),       # (24)
        speed,                    # (10)
        acc_mag[1:],              # (9)
        jerk_mag[2:],             # (8)
        speed_delta,              # (9)
        turn_cos[1:],             # (9)
        kappa[1:],                # (9)

        # ── 위치 (로컬) ──────────────────────────────────────────────────────
        traj_L.flatten(),         # (33)

        # ── 멀티스케일 CV ────────────────────────────────────────────────────
        cv_L,                     # (3)
        cv_3_L,                   # (3)
        cv_5_L,                   # (3)
        cv_all_L,                 # (3)
        cv_exp_L,                 # (3)
        cv_L - cv_5_L,            # (3)
        cv_spread,                # (3)

        # ── [NEW 1] 법선/접선 가속도 ─────────────────────────────────────────
        a_t,                      # (9)
        a_n,                      # (9)
        [a_t.mean(), a_t.std(),
         a_n.mean(), a_n.std(),
         a_n[-3:].mean(), a_n[-1]],  # (6)

        # ── [NEW 2] 다항식 외삽 ──────────────────────────────────────────────
        quad_L,                   # (3)
        cubic_L,                  # (3)
        quad_L  - cv_L,           # (3)
        cubic_L - cv_L,           # (3)

        # ── [NEW 3] 평면별 곡률 ──────────────────────────────────────────────
        kappa_xy,                 # (9)
        kappa_xz,                 # (9)
        [kappa_xy.mean(), kappa_xy.std(), kappa_xy.max(), kappa_xy[-1]],  # (4)
        [kappa_xz.mean(), kappa_xz.std(), kappa_xz.max(), kappa_xz[-1]],  # (4)

        # ── [NEW 4] 속도 R² ──────────────────────────────────────────────────
        vel_r2,                   # (3)

        # ── [NEW 5] 가속도 추세 ──────────────────────────────────────────────
        acc_slope,                # (3)

        # ── 기존 요약 통계 ───────────────────────────────────────────────────
        [turn_cos[1:].mean(), turn_cos[1:].std(),
         turn_cos[1:].min(),  turn_cos[-1]],
        [speed.mean(), speed.std(), speed[-1], speed_slope],
        [speed_delta.mean(), speed_delta.std()],
        [jerk_mag[2:].mean(), jerk_mag[2:].std(), jerk_mag[2:].max()],
        [kappa[1:].mean(), kappa[1:].std(), kappa[1:].max(), kappa[-1]],
        [acc_mag[1:].mean(), acc_mag[1:].std(), acc_mag[1:].max()],
        vel_late - vel_early,     # (3)
        vel_l2l5,                 # (3)
        vel_l3l7,                 # (3)
        last_vel_unit,            # (3)
        [cv_align],               # (1)
        [straightness],           # (1)
        disp_L,                   # (3)
        vels[-3:].mean(0) - vels.mean(0),  # (3)
        accs[1:].mean(0),         # (3)
        accs[-3:].mean(0),        # (3)
        [turn_trend],             # (1)
        [kappa_trend],            # (1)
        [speed_ratio],            # (1)
    ])
    return feats.astype(np.float32), R


def make_feature_names() -> list[str]:
    axes  = ['x', 'y', 'z']
    names: list[str] = []

    # 기존 시계열
    for s in range(10):
        for ax in axes: names.append(f"vel_{ax}_t{s}")
    for s in range(1, 10):
        for ax in axes: names.append(f"acc_{ax}_t{s}")
    for s in range(2, 10):
        for ax in axes: names.append(f"jerk_{ax}_t{s}")
    for s in range(10): names.append(f"speed_t{s}")
    for s in range(1, 10): names.append(f"acc_mag_t{s}")
    for s in range(2, 10): names.append(f"jerk_mag_t{s}")
    for s in range(9):  names.append(f"speed_delta_t{s+1}")
    for s in range(1, 10): names.append(f"turn_cos_t{s}")
    for s in range(1, 10): names.append(f"kappa_t{s}")

    # 위치
    for s in range(11):
        for ax in axes: names.append(f"pos_{ax}_t{s}")

    # 멀티스케일 CV
    for lbl in ["cv_last", "cv_3", "cv_5", "cv_all", "cv_exp",
                "cv_short_long_div", "cv_spread"]:
        for ax in axes: names.append(f"{lbl}_delta_{ax}")

    # [NEW 1] 법선/접선 가속도
    for s in range(1, 10): names.append(f"a_t_t{s}")
    for s in range(1, 10): names.append(f"a_n_t{s}")
    names += ["a_t_mean", "a_t_std", "a_n_mean", "a_n_std",
              "a_n_recent_mean", "a_n_last"]

    # [NEW 2] 다항식 외삽
    for ax in axes: names.append(f"quad_delta_{ax}")
    for ax in axes: names.append(f"cubic_delta_{ax}")
    for ax in axes: names.append(f"quad_vs_cv_{ax}")
    for ax in axes: names.append(f"cubic_vs_cv_{ax}")

    # [NEW 3] 평면별 곡률
    for s in range(1, 10): names.append(f"kappa_xy_t{s}")
    for s in range(1, 10): names.append(f"kappa_xz_t{s}")
    names += ["kappa_xy_mean", "kappa_xy_std", "kappa_xy_max", "kappa_xy_last"]
    names += ["kappa_xz_mean", "kappa_xz_std", "kappa_xz_max", "kappa_xz_last"]

    # [NEW 4] 속도 R²
    for ax in axes: names.append(f"vel_r2_{ax}")

    # [NEW 5] 가속도 추세
    for ax in axes: names.append(f"acc_slope_{ax}")

    # 기존 요약 통계
    names += ["turn_mean", "turn_std", "turn_min", "turn_last"]
    names += ["speed_mean", "speed_std", "speed_last", "speed_slope"]
    names += ["spd_delta_mean", "spd_delta_std"]
    names += ["jerk_mean", "jerk_std", "jerk_max"]
    names += ["kappa_mean", "kappa_std", "kappa_max", "kappa_last"]
    names += ["acc_mag_mean", "acc_mag_std", "acc_mag_max"]
    for ax in axes: names.append(f"vel_late_early_{ax}")
    for ax in axes: names.append(f"vel_last2_vs_last5_{ax}")
    for ax in axes: names.append(f"vel_last3_vs_last7_{ax}")
    for ax in axes: names.append(f"last_vel_unit_{ax}")
    names += ["cv_align", "straightness"]
    for ax in axes: names.append(f"disp_{ax}")
    for ax in axes: names.append(f"recent_vs_overall_{ax}")
    for ax in axes: names.append(f"mean_acc_{ax}")
    for ax in axes: names.append(f"recent_acc_{ax}")
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
    log.info("모기 비행 궤적 예측 v10 (XGBoost + FE 전면 확장 + 로컬 프레임)")
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

    residuals_train = true_xyz - cv_preds_train

    log.info("피처 생성 중...")
    train_out = [make_xgb_features(t, cv)
                 for t, cv in zip(train_data, cv_preds_train)]
    X_train   = np.array([o[0] for o in train_out])
    R_train   = np.array([o[1] for o in train_out])

    test_out  = [make_xgb_features(t, cv)
                 for t, cv in zip(test_data, cv_preds_test)]
    X_test    = np.array([o[0] for o in test_out])
    R_test    = np.array([o[1] for o in test_out])

    feat_names = make_feature_names()
    log.info(f"피처 수: {X_train.shape[1]}")
    assert X_train.shape[1] == len(feat_names), \
        f"피처 수 불일치: {X_train.shape[1]} vs {len(feat_names)}"

    # 잔차 → 로컬 프레임
    residuals_local = np.einsum('nij,nj->ni', R_train, residuals_train)

    # Train / Val 분리
    idx     = np.random.RandomState(42).permutation(len(train_data))
    val_n   = int(len(train_data) * 0.2)
    val_idx = idx[:val_n];  tr_idx = idx[val_n:]

    X_tr,  X_val  = X_train[tr_idx],         X_train[val_idx]
    y_tr,  y_val  = residuals_local[tr_idx],  residuals_local[val_idx]
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

    # Val 평가
    val_res_local  = model.predict(X_val)
    val_res_global = np.einsum('nji,nj->ni', R_val, val_res_local)
    val_preds      = val_cv + val_res_global
    log.info(f"[v10-val]   R-Hit={r_hit(val_preds, val_true):.4f}  "
             f"MeanDist={mean_dist_cm(val_preds, val_true):.2f}cm")

    tr_res_local  = model.predict(X_train)
    tr_res_global = np.einsum('nji,nj->ni', R_train, tr_res_local)
    tr_preds      = cv_preds_train + tr_res_global
    log.info(f"[v10-train] R-Hit={r_hit(tr_preds, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(tr_preds, true_xyz):.2f}cm  (과적합 참고용)")

    # Feature importance
    importances = np.array(
        [est.feature_importances_ for est in model.estimators_]
    ).mean(axis=0)
    top_idx = np.argsort(importances)[::-1][:50]
    log.info(f"\n{'='*64}")
    log.info("Top 50 피처 중요도 (x/y/z 평균)")
    log.info(f"{'='*64}")
    for rank, i in enumerate(top_idx, 1):
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        log.info(f"  {rank:2d}. {name:<40s}  {importances[i]:.4f}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    imp_df = pd.DataFrame({
        'feature':         feat_names,
        'importance_x':    model.estimators_[0].feature_importances_,
        'importance_y':    model.estimators_[1].feature_importances_,
        'importance_z':    model.estimators_[2].feature_importances_,
        'importance_mean': importances,
    }).sort_values('importance_mean', ascending=False)
    imp_df.to_csv(out_dir / "feature_importance_v10.csv", index=False)
    log.info(f"\n피처 중요도 저장 → output/feature_importance_v10.csv")

    # 제출
    test_res_local  = model.predict(X_test)
    test_res_global = np.einsum('nji,nj->ni', R_test, test_res_local)
    final_test      = cv_preds_test + test_res_global

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_test)}
    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )

    out_sub = out_dir / "submission_xgb_v10.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)

    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
