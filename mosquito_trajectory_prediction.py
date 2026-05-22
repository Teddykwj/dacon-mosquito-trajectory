import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb

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
    fh = logging.FileHandler(log_dir / "v14_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ── Data loading ───────────────────────────────────────────────────────────────
def load_sample(path: Path) -> np.ndarray:
    return pd.read_csv(path)[['x', 'y', 'z']].values


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


# ── CT (Constant Turn Rate) 물리 모델 ──────────────────────────────────────────
def predict_ct(traj: np.ndarray) -> tuple[np.ndarray, float]:
    """등속 선회율(CT) 모델: 현재 각속도 ω로 원호 경로를 예측.
    Returns (ct_pred_global, ct_weight) — ct_weight는 선회 강도 기반 신뢰도.
    """
    eps   = 1e-8
    dt    = DT * HORIZON
    vels_g = np.diff(traj, axis=0) / DT
    R      = _local_frame_rotation(vels_g[-1])
    vels   = vels_g @ R.T
    accs_r = np.diff(vels, axis=0) / DT   # (9, 3) 로컬 가속도

    speed  = float(np.linalg.norm(vels[-1]))
    v_unit = vels[-1] / (speed + eps)

    # 법선 가속도 벡터 (접선 성분 제거)
    a_t_val = float(np.dot(accs_r[-1], v_unit))
    a_n_vec = accs_r[-1] - a_t_val * v_unit
    a_n_mag = float(np.linalg.norm(a_n_vec))

    omega   = a_n_mag / (speed + eps)   # rad/s
    theta   = omega * dt                # 80ms 동안 회전각

    if omega < 1e-6 or speed < 1e-6:
        ct_L = vels[-1] * dt            # 퇴화 → CV와 동일
    else:
        R_curve = speed / (omega + eps)
        n_hat   = a_n_vec / (a_n_mag + eps)
        # 원호 위치: 전진 성분 + 법선 성분
        ct_L = R_curve * np.sin(theta) * v_unit + R_curve * (1 - np.cos(theta)) * n_hat

    ct_pred = traj[-1] + ct_L @ R      # 로컬 → 글로벌

    # 신뢰도: 회전각이 클수록 CT가 CV보다 유리 (45도에서 포화)
    ct_weight = float(np.clip(theta / (np.pi / 4), 0.0, 1.0))
    return ct_pred, ct_weight


def batch_physics_blend(data: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV + CT 적응 블렌드 앵커. Returns (blend, cv_preds, ct_weights)."""
    cv_preds  = np.array([predict_cv_last(t) for t in data])
    ct_results = [predict_ct(t) for t in data]
    ct_preds  = np.array([r[0] for r in ct_results])
    ct_weights = np.array([r[1] for r in ct_results])          # (N,)
    w = ct_weights[:, None]                                      # (N, 1) for broadcast
    blend = (1 - w) * cv_preds + w * ct_preds
    return blend, cv_preds, ct_weights


# ── Helpers ────────────────────────────────────────────────────────────────────
def _local_frame_rotation(vel: np.ndarray) -> np.ndarray:
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
    t   = np.arange(len(vels), dtype=float)
    t_c = t - t.mean()
    r2  = np.zeros(3)
    for k in range(3):
        y_c    = vels[:, k] - vels[:, k].mean()
        ss_tot = np.dot(y_c, y_c) + 1e-10
        b      = np.dot(t_c, y_c) / (np.dot(t_c, t_c) + 1e-10)
        ss_res = np.sum((y_c - b*t_c) ** 2)
        r2[k]  = max(0.0, 1.0 - ss_res / ss_tot)
    return r2


# ── Feature engineering ────────────────────────────────────────────────────────
def make_xgb_features(traj: np.ndarray,
                       cv_pred: np.ndarray,
                       ct_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    vels_g = np.diff(traj, axis=0) / DT
    R      = _local_frame_rotation(vels_g[-1])

    # 로컬 프레임 변환
    vels     = vels_g @ R.T                          # (10, 3)
    accs_raw = np.diff(vels, axis=0) / DT            # (9, 3)
    accs     = np.vstack([np.zeros((1, 3)), accs_raw])
    jerk_raw = np.diff(accs_raw, axis=0) / DT        # (8, 3)
    jerk     = np.vstack([np.zeros((2, 3)), jerk_raw])
    traj_L   = (traj - traj[-1]) @ R.T               # (11, 3)

    # 스칼라
    speed       = np.linalg.norm(vels, axis=1)       # (10,)
    acc_mag     = np.linalg.norm(accs, axis=1)
    jerk_mag    = np.linalg.norm(jerk, axis=1)
    speed_delta = np.diff(speed)                     # (9,)
    turn_cos    = _turn_cos(vels)

    kappa = np.zeros(10)
    for i in range(1, 10):
        cross    = np.cross(vels[i], accs[i])
        v_norm   = np.linalg.norm(vels[i])
        kappa[i] = np.linalg.norm(cross) / (v_norm**3 + 1e-8)

    # ── 법선/접선 가속도 ─────────────────────────────────────────────────────
    a_t = np.zeros(9)
    a_n = np.zeros(9)
    for i in range(1, 10):
        v_unit   = vels[i] / (np.linalg.norm(vels[i]) + 1e-8)
        at       = np.dot(accs[i], v_unit)
        an       = np.sqrt(max(0.0, np.dot(accs[i], accs[i]) - at**2))
        a_t[i-1] = at
        a_n[i-1] = an

    # ── [NEW] 각속도 ω = a_n / speed (선회율 rad/s) ─────────────────────────
    omega       = a_n / (speed[1:] + 1e-8)           # (9,)
    omega_trend = omega[-3:].mean() - omega[:-3].mean()

    # ── 다항식 외삽 ──────────────────────────────────────────────────────────
    t_steps = np.arange(11) * DT
    t_pred  = t_steps[-1] + HORIZON * DT
    quad_L  = np.zeros(3)
    cubic_L = np.zeros(3)
    quad_rmse  = np.zeros(3)
    cubic_rmse = np.zeros(3)
    for k in range(3):
        q2 = np.polyfit(t_steps, traj_L[:, k], 2)
        q3 = np.polyfit(t_steps, traj_L[:, k], 3)
        quad_L[k]      = np.polyval(q2, t_pred)
        cubic_L[k]     = np.polyval(q3, t_pred)
        quad_rmse[k]   = np.sqrt(np.mean((traj_L[:, k] - np.polyval(q2, t_steps))**2))
        cubic_rmse[k]  = np.sqrt(np.mean((traj_L[:, k] - np.polyval(q3, t_steps))**2))
    cv_L = vels[-1] * DT * HORIZON

    # ── [NEW] 등가속도 예측 vel[-1]*dt + 0.5*acc[-1]*dt² ────────────────────
    dt_pred   = HORIZON * DT
    ca_pred_L = vels[-1]*dt_pred + 0.5*accs[-1]*dt_pred**2   # (3,)

    # ── 기존 평면별 곡률 (크기) ───────────────────────────────────────────────
    kappa_xy = np.zeros(9)
    kappa_xz = np.zeros(9)
    for i in range(1, 10):
        v, a = vels[i], accs[i]
        c_xy = v[0]*a[1] - v[1]*a[0]
        v_xy = np.sqrt(v[0]**2 + v[1]**2)
        kappa_xy[i-1] = abs(c_xy) / (v_xy**3 + 1e-8)
        c_xz = v[0]*a[2] - v[2]*a[0]
        v_xz = np.sqrt(v[0]**2 + v[2]**2)
        kappa_xz[i-1] = abs(c_xz) / (v_xz**3 + 1e-8)

    # ── [NEW] 부호 있는 평면별 곡률 (선회 방향) ──────────────────────────────
    kappa_xy_s = np.zeros(9)   # + = 좌선회, - = 우선회
    kappa_xz_s = np.zeros(9)   # + = 상승, - = 하강 방향 선회
    for i in range(1, 10):
        v, a = vels[i], accs[i]
        c_xy = v[0]*a[1] - v[1]*a[0]
        v_xy = np.sqrt(v[0]**2 + v[1]**2)
        kappa_xy_s[i-1] = c_xy / (v_xy**3 + 1e-8)
        c_xz = v[0]*a[2] - v[2]*a[0]
        v_xz = np.sqrt(v[0]**2 + v[2]**2)
        kappa_xz_s[i-1] = c_xz / (v_xz**3 + 1e-8)

    # ── 속도 R² ──────────────────────────────────────────────────────────────
    vel_r2 = _vel_r2(vels)

    # ── 가속도 추세 ──────────────────────────────────────────────────────────
    t9        = np.arange(9, dtype=float)
    acc_slope = np.array([np.polyfit(t9, accs_raw[:, k], 1)[0] for k in range(3)])

    # ── [NEW] 속도 방향 비율 (로컬 프레임 y·z 성분) ──────────────────────────
    vel_lat_frac  = vels[:, 1] / (speed + 1e-8)     # 좌우 비율 (10,)
    vel_vert_frac = vels[:, 2] / (speed + 1e-8)     # 상하 비율 (10,)

    # ── [NEW] 선회 방향 일관성 ────────────────────────────────────────────────
    turn_dir_y = float(np.sign(vels[-3:, 1]).mean())  # -1~+1, 0=교번
    turn_dir_z = float(np.sign(vels[-3:, 2]).mean())

    # ── [NEW] a_n 급증 비율 (최근 vs 이전) ───────────────────────────────────
    a_n_ratio = a_n[-1] / (a_n[:-1].mean() + 1e-8)

    # ── 멀티스케일 CV ────────────────────────────────────────────────────────
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

    # ── [NEW] CT 모델 피처 ───────────────────────────────────────────────────
    ct_pred_global, _  = predict_ct(traj)
    ct_pred_L          = (ct_pred_global - traj[-1]) @ R.T   # 로컬 변위
    ct_vs_cv_L         = ct_pred_L - cv_L                    # CT - CV 차이

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
        traj_L.flatten(),         # (33)

        # ── 멀티스케일 CV ────────────────────────────────────────────────────
        cv_L, cv_3_L, cv_5_L, cv_all_L, cv_exp_L,  # (15)
        cv_L - cv_5_L, cv_spread,                   # (6)

        # ── 법선/접선 가속도 ─────────────────────────────────────────────────
        a_t, a_n,                 # (18)
        [a_t.mean(), a_t.std(), a_n.mean(), a_n.std(),
         a_n[-3:].mean(), a_n[-1]],  # (6)

        # ── 다항식 외삽 ──────────────────────────────────────────────────────
        quad_L, cubic_L,          # (6)
        quad_L - cv_L,            # (3)
        cubic_L - cv_L,           # (3)

        # ── 기존 평면별 곡률 (크기) ──────────────────────────────────────────
        kappa_xy, kappa_xz,       # (18)
        [kappa_xy.mean(), kappa_xy.std(), kappa_xy.max(), kappa_xy[-1]],  # (4)
        [kappa_xz.mean(), kappa_xz.std(), kappa_xz.max(), kappa_xz[-1]],  # (4)

        # ── 속도 R², 가속도 추세 ─────────────────────────────────────────────
        vel_r2, acc_slope,        # (6)

        # ── [NEW] 각속도 ω ───────────────────────────────────────────────────
        omega,                    # (9)
        [omega.mean(), omega.std(), omega[-1], omega_trend],  # (4)

        # ── [NEW] 등가속도 예측 ──────────────────────────────────────────────
        ca_pred_L,                # (3)
        ca_pred_L - cv_L,         # (3)

        # ── [NEW] 부호 있는 평면별 곡률 ─────────────────────────────────────
        kappa_xy_s, kappa_xz_s,  # (18)
        [kappa_xy_s.mean(), kappa_xy_s.std(), kappa_xy_s[-1]],  # (3)
        [kappa_xz_s.mean(), kappa_xz_s.std(), kappa_xz_s[-1]],  # (3)

        # ── [NEW] 속도 방향 비율 통계 ────────────────────────────────────────
        [vel_lat_frac.mean(),  vel_lat_frac.std(),  vel_lat_frac[-3:].mean()],  # (3)
        [vel_vert_frac.mean(), vel_vert_frac.std(), vel_vert_frac[-3:].mean()],  # (3)

        # ── [NEW] 다항식 피팅 오차 ────────────────────────────────────────────
        quad_rmse, cubic_rmse,   # (6)

        # ── [NEW] 다항식 vs cv_3 비교 ────────────────────────────────────────
        quad_L  - cv_3_L,        # (3)
        cubic_L - cv_3_L,        # (3)

        # ── [NEW] 선회 방향 일관성 / a_n 급증 ────────────────────────────────
        [turn_dir_y, turn_dir_z, a_n_ratio],  # (3)

        # ── 기존 요약 통계 ───────────────────────────────────────────────────
        [turn_cos[1:].mean(), turn_cos[1:].std(),
         turn_cos[1:].min(),  turn_cos[-1]],
        [speed.mean(), speed.std(), speed[-1], speed_slope],
        [speed_delta.mean(), speed_delta.std()],
        [jerk_mag[2:].mean(), jerk_mag[2:].std(), jerk_mag[2:].max()],
        [kappa[1:].mean(), kappa[1:].std(), kappa[1:].max(), kappa[-1]],
        [acc_mag[1:].mean(), acc_mag[1:].std(), acc_mag[1:].max()],
        vel_late - vel_early,
        vel_l2l5, vel_l3l7,
        last_vel_unit,
        [cv_align], [straightness],
        disp_L,
        vels[-3:].mean(0) - vels.mean(0),
        accs[1:].mean(0), accs[-3:].mean(0),
        [turn_trend], [kappa_trend], [speed_ratio],

        # ── [NEW] CT 모델 피처 ───────────────────────────────────────────────
        ct_pred_L,              # (3) CT 예측 로컬 변위
        ct_vs_cv_L,             # (3) CT - CV 차이 (선회 보정량)
        [ct_weight],            # (1) 선회 강도 기반 CT 신뢰도
    ])
    return feats.astype(np.float32), R


def make_feature_names() -> list[str]:
    axes  = ['x', 'y', 'z']
    N: list[str] = []

    # 기존 시계열
    for s in range(10):
        for ax in axes: N.append(f"vel_{ax}_t{s}")
    for s in range(1, 10):
        for ax in axes: N.append(f"acc_{ax}_t{s}")
    for s in range(2, 10):
        for ax in axes: N.append(f"jerk_{ax}_t{s}")
    for s in range(10): N.append(f"speed_t{s}")
    for s in range(1, 10): N.append(f"acc_mag_t{s}")
    for s in range(2, 10): N.append(f"jerk_mag_t{s}")
    for s in range(9):  N.append(f"speed_delta_t{s+1}")
    for s in range(1, 10): N.append(f"turn_cos_t{s}")
    for s in range(1, 10): N.append(f"kappa_t{s}")
    for s in range(11):
        for ax in axes: N.append(f"pos_{ax}_t{s}")

    # 멀티스케일 CV
    for lbl in ["cv_last","cv_3","cv_5","cv_all","cv_exp"]:
        for ax in axes: N.append(f"{lbl}_delta_{ax}")
    for ax in axes: N.append(f"cv_short_long_div_{ax}")
    for ax in axes: N.append(f"cv_spread_{ax}")

    # 법선/접선
    for s in range(1, 10): N.append(f"a_t_t{s}")
    for s in range(1, 10): N.append(f"a_n_t{s}")
    N += ["a_t_mean","a_t_std","a_n_mean","a_n_std","a_n_recent_mean","a_n_last"]

    # 다항식 외삽
    for ax in axes: N.append(f"quad_delta_{ax}")
    for ax in axes: N.append(f"cubic_delta_{ax}")
    for ax in axes: N.append(f"quad_vs_cv_{ax}")
    for ax in axes: N.append(f"cubic_vs_cv_{ax}")

    # 기존 평면별 곡률 (크기)
    for s in range(1, 10): N.append(f"kappa_xy_t{s}")
    for s in range(1, 10): N.append(f"kappa_xz_t{s}")
    N += ["kappa_xy_mean","kappa_xy_std","kappa_xy_max","kappa_xy_last"]
    N += ["kappa_xz_mean","kappa_xz_std","kappa_xz_max","kappa_xz_last"]

    # R², acc_slope
    for ax in axes: N.append(f"vel_r2_{ax}")
    for ax in axes: N.append(f"acc_slope_{ax}")

    # [NEW] 각속도 ω
    for s in range(1, 10): N.append(f"omega_t{s}")
    N += ["omega_mean","omega_std","omega_last","omega_trend"]

    # [NEW] 등가속도 예측
    for ax in axes: N.append(f"ca_pred_{ax}")
    for ax in axes: N.append(f"ca_vs_cv_{ax}")

    # [NEW] 부호 있는 평면별 곡률
    for s in range(1, 10): N.append(f"kappa_xy_s_t{s}")
    for s in range(1, 10): N.append(f"kappa_xz_s_t{s}")
    N += ["kappa_xy_s_mean","kappa_xy_s_std","kappa_xy_s_last"]
    N += ["kappa_xz_s_mean","kappa_xz_s_std","kappa_xz_s_last"]

    # [NEW] 속도 방향 비율
    N += ["vel_lat_mean","vel_lat_std","vel_lat_recent"]
    N += ["vel_vert_mean","vel_vert_std","vel_vert_recent"]

    # [NEW] 피팅 오차
    for ax in axes: N.append(f"quad_rmse_{ax}")
    for ax in axes: N.append(f"cubic_rmse_{ax}")

    # [NEW] 다항식 vs cv_3
    for ax in axes: N.append(f"quad_vs_cv3_{ax}")
    for ax in axes: N.append(f"cubic_vs_cv3_{ax}")

    # [NEW] 선회 일관성 / a_n 급증
    N += ["turn_dir_y","turn_dir_z","a_n_ratio"]

    # 기존 요약 통계
    N += ["turn_mean","turn_std","turn_min","turn_last"]
    N += ["speed_mean","speed_std","speed_last","speed_slope"]
    N += ["spd_delta_mean","spd_delta_std"]
    N += ["jerk_mean","jerk_std","jerk_max"]
    N += ["kappa_mean","kappa_std","kappa_max","kappa_last"]
    N += ["acc_mag_mean","acc_mag_std","acc_mag_max"]
    for ax in axes: N.append(f"vel_late_early_{ax}")
    for ax in axes: N.append(f"vel_last2_vs_last5_{ax}")
    for ax in axes: N.append(f"vel_last3_vs_last7_{ax}")
    for ax in axes: N.append(f"last_vel_unit_{ax}")
    N += ["cv_align","straightness"]
    for ax in axes: N.append(f"disp_{ax}")
    for ax in axes: N.append(f"recent_vs_overall_{ax}")
    for ax in axes: N.append(f"mean_acc_{ax}")
    for ax in axes: N.append(f"recent_acc_{ax}")
    N += ["turn_trend","kappa_trend","speed_ratio"]

    # [NEW] CT 모델 피처
    for ax in axes: N.append(f"ct_pred_{ax}")
    for ax in axes: N.append(f"ct_vs_cv_{ax}")
    N += ["ct_weight"]
    return N


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1) <= 0.01))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 66)
    log.info("모기 비행 궤적 예측 v14 (XGBoost + LightGBM 5-Fold 앙상블)")
    log.info("=" * 66)

    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)
    log.info(f"Train: {len(train_data)}개  |  Test: {len(test_data)}개")

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    # ── 물리 모델 앵커 계산 ────────────────────────────────────────────────────
    cv_preds_train = batch_cv_last(train_data)
    cv_preds_test  = batch_cv_last(test_data)
    log.info(f"[CV-last]   R-Hit={r_hit(cv_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_preds_train, true_xyz):.2f}cm")

    blend_train, _, ct_w_train = batch_physics_blend(train_data)
    blend_test,  _, _          = batch_physics_blend(test_data)
    log.info(f"[CT-blend]  R-Hit={r_hit(blend_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(blend_train, true_xyz):.2f}cm  "
             f"(CT weight 평균: {ct_w_train.mean():.3f})")

    residuals_train = true_xyz - blend_train   # 블렌드 앵커 기준 잔차

    log.info("피처 생성 중...")
    train_out = [make_xgb_features(t, cv, cw)
                 for t, cv, cw in zip(train_data, cv_preds_train, ct_w_train)]
    X_train   = np.array([o[0] for o in train_out])
    R_train   = np.array([o[1] for o in train_out])

    ct_w_test = np.array([predict_ct(t)[1] for t in test_data])
    test_out  = [make_xgb_features(t, cv, cw)
                 for t, cv, cw in zip(test_data, cv_preds_test, ct_w_test)]
    X_test    = np.array([o[0] for o in test_out])
    R_test    = np.array([o[1] for o in test_out])

    feat_names = make_feature_names()
    log.info(f"피처 수: {X_train.shape[1]}")
    assert X_train.shape[1] == len(feat_names), \
        f"피처 불일치: {X_train.shape[1]} vs {len(feat_names)}"

    residuals_local = np.einsum('nij,nj->ni', R_train, residuals_train)

    N_FOLDS       = 5
    kf            = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_xgb       = np.zeros_like(true_xyz)   # XGBoost OOF 잔차 (글로벌)
    oof_lgb       = np.zeros_like(true_xyz)   # LightGBM OOF 잔차 (글로벌)
    test_xgb_acc  = np.zeros((len(test_data), 3))
    test_lgb_acc  = np.zeros((len(test_data), 3))
    imp_xgb_acc   = np.zeros(len(feat_names))
    imp_lgb_acc   = np.zeros(len(feat_names))

    log.info("5-Fold XGBoost + LightGBM 앙상블 학습 시작...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr  = X_train[tr_idx];  X_val = X_train[val_idx]
        y_tr  = residuals_local[tr_idx]
        R_val = R_train[val_idx]
        val_blend_f = blend_train[val_idx]
        val_true_f  = true_xyz[val_idx]

        # ── XGBoost ──────────────────────────────────────────────────────────
        xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0,
        ), n_jobs=1)
        xgb_model.fit(X_tr, y_tr)

        xgb_val_local  = xgb_model.predict(X_val)
        xgb_val_global = np.einsum('nji,nj->ni', R_val, xgb_val_local)
        oof_xgb[val_idx] = xgb_val_global

        xgb_test_local  = xgb_model.predict(X_test)
        test_xgb_acc   += np.einsum('nji,nj->ni', R_test, xgb_test_local)
        imp_xgb_acc    += np.array([e.feature_importances_
                                    for e in xgb_model.estimators_]).mean(0)

        # ── LightGBM ─────────────────────────────────────────────────────────
        lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            random_state=42, n_jobs=-1, verbosity=-1,
        ), n_jobs=1)
        lgb_model.fit(X_tr, y_tr)

        lgb_val_local  = lgb_model.predict(X_val)
        lgb_val_global = np.einsum('nji,nj->ni', R_val, lgb_val_local)
        oof_lgb[val_idx] = lgb_val_global

        lgb_test_local  = lgb_model.predict(X_test)
        test_lgb_acc   += np.einsum('nji,nj->ni', R_test, lgb_test_local)
        imp_lgb_acc    += np.array([e.feature_importances_
                                    for e in lgb_model.estimators_]).mean(0)

        # ── Fold 점수 ────────────────────────────────────────────────────────
        oof_blend_f = val_blend_f + 0.5 * (xgb_val_global + lgb_val_global)
        xgb_hit = r_hit(val_blend_f + xgb_val_global, val_true_f)
        lgb_hit = r_hit(val_blend_f + lgb_val_global, val_true_f)
        ens_hit = r_hit(oof_blend_f, val_true_f)
        log.info(f"  Fold {fold}/{N_FOLDS}  XGB={xgb_hit:.4f}  "
                 f"LGB={lgb_hit:.4f}  Ensemble={ens_hit:.4f}")

    # ── 앙상블 최종 예측 ──────────────────────────────────────────────────────
    avg_xgb_res = test_xgb_acc / N_FOLDS
    avg_lgb_res = test_lgb_acc / N_FOLDS
    final_test  = blend_test + 0.5 * (avg_xgb_res + avg_lgb_res)

    oof_preds = blend_train + 0.5 * (oof_xgb + oof_lgb)
    oof_xgb_p = blend_train + oof_xgb
    oof_lgb_p = blend_train + oof_lgb
    log.info(f"\n[v14-OOF-XGB] R-Hit={r_hit(oof_xgb_p, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(oof_xgb_p, true_xyz):.2f}cm")
    log.info(f"[v14-OOF-LGB] R-Hit={r_hit(oof_lgb_p, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(oof_lgb_p, true_xyz):.2f}cm")
    log.info(f"[v14-OOF-ENS] R-Hit={r_hit(oof_preds, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(oof_preds, true_xyz):.2f}cm")

    importances = (imp_xgb_acc + imp_lgb_acc) / (2 * N_FOLDS)
    top_idx = np.argsort(importances)[::-1][:50]
    log.info(f"\n{'='*66}")
    log.info("Top 50 피처 중요도 (XGB+LGB 5-fold 평균)")
    log.info(f"{'='*66}")
    for rank, i in enumerate(top_idx, 1):
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        log.info(f"  {rank:2d}. {name:<42s}  {importances[i]:.4f}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({
        'feature':         feat_names,
        'importance_mean': importances,
    }).sort_values('importance_mean', ascending=False).to_csv(
        out_dir / "feature_importance_v14.csv", index=False)
    log.info(f"\n피처 중요도 저장 → output/feature_importance_v14.csv")

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_test)}
    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )
    out_sub = out_dir / "submission_xgb_v14.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)
    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
