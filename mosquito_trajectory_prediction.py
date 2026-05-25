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
    fh = logging.FileHandler(log_dir / "v19_log.txt", encoding="utf-8")
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


# ── 3D 회전 증강 ───────────────────────────────────────────────────────────────
def _random_rotation(rng: np.random.RandomState) -> np.ndarray:
    """SO(3) 균일 분포 무작위 회전행렬 (QR 분해 방식)."""
    H = rng.randn(3, 3)
    Q, R = np.linalg.qr(H)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


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


# ── Error analysis ────────────────────────────────────────────────────────────
def compute_traj_stats(traj: np.ndarray, ct_weight: float) -> dict:
    """궤적 하나에서 해석 가능한 통계를 계산."""
    eps = 1e-8
    vels_g = np.diff(traj, axis=0) / DT          # (10, 3) 글로벌 속도
    speed  = np.linalg.norm(vels_g, axis=1)       # (10,)

    # 곡률
    accs_g = np.diff(vels_g, axis=0) / DT         # (9, 3)
    kappa_last = 0.0
    if speed[-1] > eps:
        cross = np.cross(vels_g[-1], accs_g[-1])
        kappa_last = float(np.linalg.norm(cross) / (speed[-1]**3 + eps))

    # 마지막 방향 전환
    turn_cos_last = 1.0
    if speed[-1] > eps and speed[-2] > eps:
        turn_cos_last = float(np.dot(vels_g[-1], vels_g[-2]) /
                              (speed[-1] * speed[-2] + eps))

    # 속력 추세 (양수=가속, 음수=감속)
    speed_trend = float(np.polyfit(np.arange(10), speed, 1)[0])

    path_len     = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    straightness = float(np.linalg.norm(traj[-1] - traj[0]) / (path_len + eps))

    return {
        'speed_last':    float(speed[-1]),
        'speed_mean':    float(speed.mean()),
        'speed_trend':   speed_trend,
        'kappa_last':    kappa_last,
        'turn_cos_last': turn_cos_last,
        'straightness':  straightness,
        'ct_weight':     ct_weight,
    }


def run_error_analysis(log, train_data, true_xyz, oof_preds,
                       cv_preds_train, blend_train, ct_w_train,
                       train_ids, out_dir):
    """OOF 에러 분석 — 궤적 패턴별 실패 원인 분해."""
    n = len(train_data)
    errors_cm = np.linalg.norm(oof_preds - true_xyz, axis=1) * 100
    cv_errs   = np.linalg.norm(cv_preds_train - true_xyz, axis=1) * 100
    bl_errs   = np.linalg.norm(blend_train    - true_xyz, axis=1) * 100
    hits      = errors_cm <= 1.0

    # 궤적 통계 계산
    stats_rows = [compute_traj_stats(train_data[i], float(ct_w_train[i]))
                  for i in range(n)]
    df = pd.DataFrame(stats_rows)
    df['id']          = train_ids
    df['error_cm']    = errors_cm
    df['cv_error_cm'] = cv_errs
    df['bl_error_cm'] = bl_errs
    df['hit']         = hits.astype(int)
    df['improvement_vs_cv']    = cv_errs  - errors_cm   # 양수 = 개선
    df['improvement_vs_blend'] = bl_errs  - errors_cm

    # ── 저장 ──────────────────────────────────────────────────────────────────
    save_path = out_dir / "oof_analysis_v19.csv"
    df.to_csv(save_path, index=False)
    log.info(f"\nOOF 분석 저장 → {save_path}")

    # ── 에러 분포 ─────────────────────────────────────────────────────────────
    pcts = np.percentile(errors_cm, [25, 50, 75, 90, 95, 99])
    log.info("\n[에러 분포 (cm)]")
    log.info(f"  p25={pcts[0]:.2f}  p50={pcts[1]:.2f}  p75={pcts[2]:.2f}"
             f"  p90={pcts[3]:.2f}  p95={pcts[4]:.2f}  p99={pcts[5]:.2f}")
    log.info(f"  R-Hit@1cm={hits.mean():.4f}  (mean={errors_cm.mean():.2f}cm"
             f"  max={errors_cm.max():.2f}cm)")

    # ── 속력 구간별 분석 ──────────────────────────────────────────────────────
    log.info("\n[속력 구간별 R-Hit]  (speed_last 기준 5분위)")
    speed_q = pd.qcut(df['speed_last'], 5, labels=['Q1(느림)','Q2','Q3','Q4','Q5(빠름)'])
    for grp, sub in df.groupby(speed_q, observed=True):
        log.info(f"  {grp}: R-Hit={sub['hit'].mean():.3f}  "
                 f"mean_err={sub['error_cm'].mean():.2f}cm  n={len(sub)}")

    # ── 곡률 구간별 분석 ──────────────────────────────────────────────────────
    log.info("\n[곡률 구간별 R-Hit]  (kappa_last 기준 5분위)")
    kappa_q = pd.qcut(df['kappa_last'], 5, labels=['Q1(직진)','Q2','Q3','Q4','Q5(급선회)'])
    for grp, sub in df.groupby(kappa_q, observed=True):
        log.info(f"  {grp}: R-Hit={sub['hit'].mean():.3f}  "
                 f"mean_err={sub['error_cm'].mean():.2f}cm  n={len(sub)}")

    # ── CT weight 구간별 분석 ─────────────────────────────────────────────────
    log.info("\n[CT weight 구간별 R-Hit]  (선회 강도)")
    ct_bins  = [0.0, 0.1, 0.3, 0.6, 1.01]
    ct_labels = ['직진(0~0.1)', '약선회(0.1~0.3)', '중선회(0.3~0.6)', '강선회(0.6~1.0)']
    ct_q = pd.cut(df['ct_weight'], bins=ct_bins, labels=ct_labels, include_lowest=True)
    for grp, sub in df.groupby(ct_q, observed=True):
        log.info(f"  {grp}: R-Hit={sub['hit'].mean():.3f}  "
                 f"mean_err={sub['error_cm'].mean():.2f}cm  n={len(sub)}")

    # ── 속력 추세별 분석 (가속/감속) ─────────────────────────────────────────
    log.info("\n[속력 추세별 R-Hit]")
    trend_q = pd.qcut(df['speed_trend'], 3, labels=['감속','등속','가속'])
    for grp, sub in df.groupby(trend_q, observed=True):
        log.info(f"  {grp}: R-Hit={sub['hit'].mean():.3f}  "
                 f"mean_err={sub['error_cm'].mean():.2f}cm  n={len(sub)}")

    # ── CV-last 대비 개선/악화 분석 ───────────────────────────────────────────
    improved  = df['improvement_vs_cv'] > 0.05
    worsened  = df['improvement_vs_cv'] < -0.05
    log.info(f"\n[CV-last 대비]")
    log.info(f"  개선된 샘플: {improved.sum()}개 ({improved.mean()*100:.1f}%)  "
             f"평균 개선량={df.loc[improved,'improvement_vs_cv'].mean():.2f}cm")
    log.info(f"  악화된 샘플: {worsened.sum()}개 ({worsened.mean()*100:.1f}%)  "
             f"평균 악화량={-df.loc[worsened,'improvement_vs_cv'].mean():.2f}cm")

    # ── 최악 샘플 특성 ────────────────────────────────────────────────────────
    top_bad = df.nlargest(200, 'error_cm')
    log.info(f"\n[최악 200개 샘플 평균 특성]  (error_cm 기준)")
    log.info(f"  speed_last={top_bad['speed_last'].mean():.2f}  "
             f"kappa_last={top_bad['kappa_last'].mean():.4f}  "
             f"ct_weight={top_bad['ct_weight'].mean():.3f}")
    log.info(f"  straightness={top_bad['straightness'].mean():.3f}  "
             f"turn_cos_last={top_bad['turn_cos_last'].mean():.3f}")
    log.info(f"  전체 대비: speed={df['speed_last'].mean():.2f}  "
             f"kappa={df['kappa_last'].mean():.4f}  "
             f"ct_w={df['ct_weight'].mean():.3f}")


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1) <= 0.01))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Group training ────────────────────────────────────────────────────────────
def train_group(log, label,
                train_data_g, X_g, R_g,
                blend_g, true_xyz_g, cv_preds_g, ct_w_g, disp_scale_g,
                X_test, R_test, disp_scale_test,
                feat_names, N_AUG=4, n_folds=5):
    """속력 그룹별 독립 증강 + 5-Fold 학습."""
    n = len(train_data_g)
    rng = np.random.RandomState(123)

    all_X     = [X_g];         all_R     = [R_g]
    all_blend = [blend_g];     all_true  = [true_xyz_g]
    all_scale = [disp_scale_g]

    for _ in range(N_AUG):
        Q_batch   = np.array([_random_rotation(rng) for _ in range(n)])
        trajs_rot = [(Q_batch[i] @ train_data_g[i].T).T for i in range(n)]
        true_rot  = np.einsum('nij,nj->ni', Q_batch, true_xyz_g)
        blend_rot = np.einsum('nij,nj->ni', Q_batch, blend_g)
        cv_rot    = np.einsum('nij,nj->ni', Q_batch, cv_preds_g)
        aug_out   = [make_xgb_features(t, cv, cw)
                     for t, cv, cw in zip(trajs_rot, cv_rot, ct_w_g)]
        all_X.append(np.array([o[0] for o in aug_out]))
        all_R.append(np.array([o[1] for o in aug_out]))
        all_blend.append(blend_rot)
        all_true.append(true_rot)
        all_scale.append(disp_scale_g)

    X_all          = np.concatenate(all_X,     axis=0)
    R_all          = np.concatenate(all_R,     axis=0)
    blend_all_aug  = np.concatenate(all_blend, axis=0)
    true_all       = np.concatenate(all_true,  axis=0)
    disp_scale_all = np.concatenate(all_scale, axis=0)
    res_loc_norm   = (np.einsum('nij,nj->ni', R_all, true_all - blend_all_aug)
                      / disp_scale_all[:, None])

    kf           = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof          = np.zeros_like(true_xyz_g)
    test_res_acc = np.zeros((len(disp_scale_test), 3))
    imp_acc      = np.zeros(len(feat_names))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n)), 1):
        tr_aug_idx = np.concatenate([tr_idx + n * r for r in range(N_AUG + 1)])
        model = MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0,
        ), n_jobs=1)
        model.fit(X_all[tr_aug_idx], res_loc_norm[tr_aug_idx])

        val_res_local  = model.predict(X_all[val_idx]) * disp_scale_g[val_idx, None]
        val_res_global = np.einsum('nji,nj->ni', R_all[val_idx], val_res_local)
        oof[val_idx]   = blend_all_aug[val_idx] + val_res_global
        log.info(f"  [{label}] Fold {fold}/{n_folds}  "
                 f"R-Hit={r_hit(oof[val_idx], true_xyz_g[val_idx]):.4f}  "
                 f"(n_tr={len(tr_aug_idx):,})")

        test_res_local  = model.predict(X_test) * disp_scale_test[:, None]
        test_res_acc   += np.einsum('nji,nj->ni', R_test, test_res_local)
        imp_acc        += np.array([e.feature_importances_
                                    for e in model.estimators_]).mean(0)

    return oof, test_res_acc / n_folds, imp_acc / n_folds


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 66)
    log.info("모기 비행 궤적 예측 v19 (v17 + 메타 게이팅)")
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

    # ── 속력 기반 잔차 정규화 스케일 ──────────────────────────────────────────
    # 빠른 샘플과 느린 샘플의 잔차 크기를 통일 → 학습 신호 균형
    # speed * dt = 예상 이동거리 (cm), 회전에 불변 → 증강에서도 재사용 가능
    speed_train = np.array([np.linalg.norm((t[-1] - t[-2]) / DT) for t in train_data])
    disp_scale_train = np.maximum(speed_train * DT * HORIZON, 0.01)   # (N,) cm

    speed_test = np.array([np.linalg.norm((t[-1] - t[-2]) / DT) for t in test_data])
    disp_scale_test = np.maximum(speed_test * DT * HORIZON, 0.01)     # (N,) cm

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

    # ── 단일 모델 5-Fold 학습 (v17 구조 복귀) ────────────────────────────────
    log.info("5-Fold XGBoost 학습 시작 (N_AUG=4)...")
    oof_preds, xgb_test_res, imp_acc = train_group(
        log, "All",
        train_data, X_train, R_train,
        blend_train, true_xyz,
        cv_preds_train, ct_w_train, disp_scale_train,
        X_test, R_test, disp_scale_test,
        feat_names, N_AUG=4,
    )
    log.info(f"\n[v19-OOF]  R-Hit={r_hit(oof_preds, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(oof_preds, true_xyz):.2f}cm")

    # ── 메타 게이팅: XGBoost 보정이 도움되는 샘플만 선별 ──────────────────────
    oof_err   = np.linalg.norm(oof_preds - true_xyz, axis=1)
    blend_err = np.linalg.norm(blend_train - true_xyz, axis=1)
    gate_labels = (oof_err < blend_err).astype(int)   # 1=개선, 0=악화
    log.info(f"\n[Gate] 개선={gate_labels.sum()}개  악화={(~gate_labels.astype(bool)).sum()}개  "
             f"개선율={gate_labels.mean()*100:.1f}%")

    gate_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        tree_method='hist', random_state=42, n_jobs=-1, verbosity=0,
    )
    gate_clf.fit(X_train, gate_labels)

    gate_prob_test  = gate_clf.predict_proba(X_test)[:, 1]    # (N_test,)
    gate_prob_train = gate_clf.predict_proba(X_train)[:, 1]   # (N_train, in-sample)

    # in-sample 게이팅 효과 (낙관적이지만 방향성 확인용)
    xgb_train_res = oof_preds - blend_train
    oof_gated     = blend_train + gate_prob_train[:, None] * xgb_train_res
    log.info(f"[OOF-Gated] R-Hit={r_hit(oof_gated, true_xyz):.4f}  "
             f"(raw OOF={r_hit(oof_preds, true_xyz):.4f})")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    run_error_analysis(log, train_data, true_xyz, oof_preds,
                       cv_preds_train, blend_train, ct_w_train,
                       train_ids, out_dir)

    # ── 테스트 최종 예측: soft gating 적용 ───────────────────────────────────
    final_test = blend_test + gate_prob_test[:, None] * xgb_test_res

    importances = imp_acc
    top_idx = np.argsort(importances)[::-1][:50]
    log.info(f"\n{'='*66}")
    log.info("Top 50 피처 중요도 (5-fold 평균)")
    log.info(f"{'='*66}")
    for rank, i in enumerate(top_idx, 1):
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        log.info(f"  {rank:2d}. {name:<42s}  {importances[i]:.4f}")

    pd.DataFrame({
        'feature':         feat_names,
        'importance_mean': importances,
    }).sort_values('importance_mean', ascending=False).to_csv(
        out_dir / "feature_importance_v19.csv", index=False)
    log.info(f"\n피처 중요도 저장 → output/feature_importance_v19.csv")

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_test)}
    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )
    out_sub = out_dir / "submission_xgb_v19.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)
    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
