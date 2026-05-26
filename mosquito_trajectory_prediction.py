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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
    fh = logging.FileHandler(log_dir / "v25_log.txt", encoding="utf-8")
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


# ── Kalman 필터 ───────────────────────────────────────────────────────────────
def _kalman_last_state(traj: np.ndarray,
                       q_acc: float = 10.0,
                       r_pos: float = 0.002) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CA Kalman forward filter (11 관측값 → 마지막 프레임 pos/vel/acc 추정).
    q_acc: 가속도 변화 노이즈 표준편차 (m/s² per frame)
    r_pos: 위치 측정 노이즈 표준편차 (m)
    """
    n  = len(traj)   # 11
    dt = DT
    F  = np.array([[1., dt, 0.5 * dt * dt],
                   [0., 1., dt           ],
                   [0., 0., 1.           ]])
    Q  = np.zeros((3, 3))
    Q[2, 2] = q_acc * q_acc          # 가속도만 랜덤 변화
    r2 = r_pos * r_pos

    out = np.zeros((3, 3))           # out[d] = [pos, vel, acc] for dim d
    for d in range(3):
        x = np.array([traj[0, d],
                      (traj[1, d] - traj[0, d]) / dt,
                      0.0])
        P = np.diag([r2,
                     2.0 * r2 / (dt * dt),   # 2-점 속도 추정 불확실성
                     (q_acc * 4.0) ** 2])     # 가속도 초기 불확실성 높게
        for k in range(1, n):
            x      = F @ x
            P      = F @ P @ F.T + Q
            innov  = traj[k, d] - x[0]
            s_val  = P[0, 0] + r2
            K      = P[:, 0] / s_val          # (3,) Kalman gain
            x      = x + K * innov
            P      = P - np.outer(K, P[0, :]) # (I - KH)P
        out[d] = x
    return out[:, 0], out[:, 1], out[:, 2]    # pos(3), vel(3), acc(3)


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


# ── CA (Constant Acceleration) Kalman 앵커 ────────────────────────────────────
def predict_ca_kf(traj: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Kalman 스무딩 기반 CA 앵커.
    Returns (ca_pred_global, ca_weight, cv_smooth_pred).
    ca_weight: 접선 가속도 기반 [0, 0.5]
    cv_smooth_pred: 스무딩된 속도로 계산한 CV (noise 감소)
    """
    eps     = 1e-8
    dt_pred = DT * HORIZON
    smooth_pos, smooth_vel, smooth_acc = _kalman_last_state(traj)

    ca_pred   = smooth_pos + smooth_vel * dt_pred + 0.5 * smooth_acc * dt_pred ** 2
    cv_smooth = smooth_pos + smooth_vel * dt_pred

    speed = float(np.linalg.norm(smooth_vel))
    v_hat = smooth_vel / (speed + eps)
    a_t   = float(np.dot(smooth_acc, v_hat))   # 접선 가속도 (속력 변화)
    # CA-CV 차이 / 예상 이동거리 = 0.5*|a_t|*dt² / (speed*dt) = 0.5*|a_t|*dt/speed
    ca_weight = float(np.clip(0.5 * abs(a_t) * dt_pred / (speed + eps), 0.0, 0.5))
    return ca_pred, ca_weight, cv_smooth


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
                       ct_weight: float = 0.0,
                       cv_smooth: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
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

        # ── [NEW] CV-smooth 피처 (Kalman 스무딩 속도 기반) ──────────────────────
        *(((cv_smooth - traj[-1]) @ R.T,                    # cv_smooth_L (3)
           (cv_smooth - traj[-1]) @ R.T - cv_L,             # cv_smooth_vs_raw_L (3)
           ) if cv_smooth is not None else
          (np.zeros(3), np.zeros(3))),
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

    # [NEW] CV-smooth 피처
    for ax in axes: N.append(f"cv_smooth_{ax}")
    for ax in axes: N.append(f"cv_smooth_vs_raw_{ax}")
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
    save_path = out_dir / "oof_analysis_v25.csv"
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
                feat_names,
                cv_smooth_g=None, cv_smooth_test=None,
                N_AUG=4, n_folds=5):
    """5-Fold 학습 + 3D 회전 증강."""
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
        cv_sm_rot = (np.einsum('nij,nj->ni', Q_batch, cv_smooth_g)
                     if cv_smooth_g is not None else [None] * n)
        aug_out   = [make_xgb_features(t, cv, cw, cvs)
                     for t, cv, cw, cvs in zip(trajs_rot, cv_rot, ct_w_g, cv_sm_rot)]
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


# ── DL: 입력 준비 ─────────────────────────────────────────────────────────────
def prepare_dl_inputs(traj: np.ndarray,
                      ct_pred_global: np.ndarray,
                      ct_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """로컬 프레임 속도 시퀀스 + 물리 피처.
    Returns (vel_seq [10,3], phys [7], R [3,3]).
    """
    vels_g    = np.diff(traj, axis=0) / DT               # (10, 3) global
    R         = _local_frame_rotation(vels_g[-1])
    vel_seq   = (vels_g @ R.T).astype(np.float32)        # (10, 3) local frame
    cv_L      = vels_g[-1] @ R.T * DT * HORIZON          # (3,) CV local
    ct_pred_L = ((ct_pred_global - traj[-1]) @ R.T).astype(np.float32)
    ct_vs_cv_L = (ct_pred_L - cv_L).astype(np.float32)
    phys = np.concatenate([ct_pred_L, ct_vs_cv_L, [ct_weight]]).astype(np.float32)
    return vel_seq, phys, R


# ── DL: Transformer 모델 ───────────────────────────────────────────────────────
class TrajTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vel_embed = nn.Linear(3, d_model)
        self.pos_enc   = nn.Parameter(torch.zeros(10, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + 7, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, vel_seq: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        x = self.vel_embed(vel_seq) + self.pos_enc   # (B, 10, d_model)
        x = self.encoder(x)[:, -1]                   # (B, d_model) — last token
        return self.head(torch.cat([x, phys], dim=-1))


# ── DL: 5-Fold 학습 ───────────────────────────────────────────────────────────
def train_dl_5fold(log,
                   train_data, blend_train, true_xyz,
                   ct_results_train, disp_scale_train,
                   test_data, blend_test, ct_results_test, disp_scale_test,
                   N_AUG: int = 4, n_folds: int = 5,
                   n_epochs: int = 50, batch_size: int = 256):
    """5-Fold Transformer 학습 + 3D 회전 증강 (속력 기반 정규화)."""
    n  = len(train_data)
    nt = len(test_data)
    rng = np.random.RandomState(123)

    def _prep(data, ct_res, disp_sc):
        outs = [prepare_dl_inputs(t, r[0], r[1]) for t, r in zip(data, ct_res)]
        vel  = np.array([o[0] for o in outs])   # (N, 10, 3)
        phy  = np.array([o[1] for o in outs])   # (N, 7)
        Rm   = np.array([o[2] for o in outs])   # (N, 3, 3)
        # 속력 기반 정규화 → 스케일 불변
        vel  = (vel * (DT * HORIZON) / disp_sc[:, None, None]).astype(np.float32)
        phy2 = phy.copy()
        phy2[:, :6] /= disp_sc[:, None]         # ct_pred_L, ct_vs_cv_L → 무차원
        return vel, phy2.astype(np.float32), Rm

    vel_base, phys_base, R_base = _prep(train_data, ct_results_train, disp_scale_train)
    vel_test, phys_test, R_test = _prep(test_data,  ct_results_test,  disp_scale_test)

    # 베이스 타겟: R @ (true - blend) / disp_scale
    res_base = true_xyz - blend_train
    tgt_base = (np.einsum('nij,nj->ni', R_base, res_base)
                / disp_scale_train[:, None]).astype(np.float32)

    # 증강 데이터 구성
    all_vel  = [vel_base]
    all_phys = [phys_base]
    all_tgt  = [tgt_base]

    for _ in range(N_AUG):
        Q_batch  = np.array([_random_rotation(rng) for _ in range(n)])
        vel_aug  = np.zeros_like(vel_base)
        phys_aug = np.zeros_like(phys_base)
        tgt_aug  = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            Q  = Q_batch[i]
            tq = (Q @ train_data[i].T).T
            cp = Q @ ct_results_train[i][0]
            cw = ct_results_train[i][1]
            vs, ph, Rq = prepare_dl_inputs(tq, cp, cw)
            ds = disp_scale_train[i]
            vel_aug[i]  = (vs * (DT * HORIZON) / ds).astype(np.float32)
            ph2 = ph.copy(); ph2[:6] /= ds
            phys_aug[i] = ph2.astype(np.float32)
            res_Q = Q @ (true_xyz[i] - blend_train[i])
            tgt_aug[i]  = (Rq @ res_Q / ds).astype(np.float32)
        all_vel.append(vel_aug)
        all_phys.append(phys_aug)
        all_tgt.append(tgt_aug)

    vel_all = np.concatenate(all_vel,  axis=0)   # (N*(N_AUG+1), 10, 3)
    phy_all = np.concatenate(all_phys, axis=0)
    tgt_all = np.concatenate(all_tgt,  axis=0)

    Vt = torch.from_numpy(vel_test)
    Pt = torch.from_numpy(phys_test)

    kf           = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds    = np.zeros_like(true_xyz)
    test_res_acc = np.zeros((nt, 3), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n)), 1):
        tr_aug_idx = np.concatenate([tr_idx + n * r for r in range(N_AUG + 1)])
        X_tr  = torch.from_numpy(vel_all[tr_aug_idx])
        P_tr  = torch.from_numpy(phy_all[tr_aug_idx])
        Y_tr  = torch.from_numpy(tgt_all[tr_aug_idx])
        X_val = torch.from_numpy(vel_base[val_idx])
        P_val = torch.from_numpy(phys_base[val_idx])

        ds_tr  = TensorDataset(X_tr, P_tr, Y_tr)
        dl_tr  = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

        model = TrajTransformer()
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        crit  = nn.MSELoss()

        last_loss = 0.0
        for epoch in range(n_epochs):
            model.train()
            ep_loss = 0.0
            for Xb, Pb, Yb in dl_tr:
                pred  = model(Xb, Pb)
                loss  = crit(pred, Yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            sched.step()
            last_loss = ep_loss / len(dl_tr)

        model.eval()
        with torch.no_grad():
            val_res_norm = model(X_val, P_val).numpy()
        val_res_local  = val_res_norm * disp_scale_train[val_idx, None]
        val_res_global = np.einsum('nji,nj->ni', R_base[val_idx], val_res_local)
        oof_preds[val_idx] = blend_train[val_idx] + val_res_global
        log.info(f"  [DL] Fold {fold}/{n_folds}  "
                 f"R-Hit={r_hit(oof_preds[val_idx], true_xyz[val_idx]):.4f}  "
                 f"loss={last_loss:.5f}  (n_tr={len(tr_aug_idx):,})")

        with torch.no_grad():
            test_res_norm = model(Vt, Pt).numpy()
        test_res_local  = test_res_norm * disp_scale_test[:, None]
        test_res_global = np.einsum('nji,nj->ni', R_test, test_res_local)
        test_res_acc   += test_res_global

    dl_test_res = test_res_acc / n_folds   # (nt, 3) 평균 잔차 (global)
    return oof_preds, dl_test_res


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 66)
    log.info("모기 비행 궤적 예측 v25 (CV+CT 블렌드 + cv_smooth 피처 + XGBoost 게이팅)")
    log.info("=" * 66)

    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)
    log.info(f"Train: {len(train_data)}개  |  Test: {len(test_data)}개")

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    # ── 1. CV 기준선 ─────────────────────────────────────────────────────────
    cv_preds_train = batch_cv_last(train_data)
    cv_preds_test  = batch_cv_last(test_data)
    log.info(f"[CV-last]   R-Hit={r_hit(cv_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_preds_train, true_xyz):.2f}cm")

    # ── 2. CT 모델 ───────────────────────────────────────────────────────────
    ct_results_train = [predict_ct(t) for t in train_data]
    ct_preds_train   = np.array([r[0] for r in ct_results_train])
    ct_w_train       = np.array([r[1] for r in ct_results_train])

    ct_results_test  = [predict_ct(t) for t in test_data]
    ct_preds_test    = np.array([r[0] for r in ct_results_test])
    ct_w_test        = np.array([r[1] for r in ct_results_test])

    # ── 3. 2-way 블렌드 앵커 (CV + CT) ──────────────────────────────────────
    w_tr          = ct_w_train[:, None]
    blend_train   = (1 - w_tr) * cv_preds_train + w_tr * ct_preds_train
    w_te          = ct_w_test[:, None]
    blend_test    = (1 - w_te) * cv_preds_test  + w_te * ct_preds_test
    log.info(f"[CT-blend]  R-Hit={r_hit(blend_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(blend_train, true_xyz):.2f}cm  "
             f"(CT weight 평균: {ct_w_train.mean():.3f})")

    # ── 4. Kalman 스무딩 → cv_smooth 피처 ───────────────────────────────────
    log.info("Kalman 스무딩 계산 중...")
    ka_results_train = [predict_ca_kf(t) for t in train_data]
    cv_smooth_train  = np.array([r[2] for r in ka_results_train])

    ka_results_test  = [predict_ca_kf(t) for t in test_data]
    cv_smooth_test   = np.array([r[2] for r in ka_results_test])
    log.info(f"[CV-smooth] R-Hit={r_hit(cv_smooth_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_smooth_train, true_xyz):.2f}cm")

    # ── 5. 속력 기반 정규화 스케일 ──────────────────────────────────────────
    speed_train      = np.array([np.linalg.norm((t[-1] - t[-2]) / DT) for t in train_data])
    disp_scale_train = np.maximum(speed_train * DT * HORIZON, 0.01)

    speed_test       = np.array([np.linalg.norm((t[-1] - t[-2]) / DT) for t in test_data])
    disp_scale_test  = np.maximum(speed_test  * DT * HORIZON, 0.01)

    # ── 6. XGBoost 피처 ─────────────────────────────────────────────────────
    log.info("피처 생성 중...")
    train_out = [make_xgb_features(t, cv, cw, cvs)
                 for t, cv, cw, cvs in zip(
                     train_data, cv_preds_train, ct_w_train, cv_smooth_train)]
    X_train   = np.array([o[0] for o in train_out])
    R_train   = np.array([o[1] for o in train_out])

    test_out  = [make_xgb_features(t, cv, cw, cvs)
                 for t, cv, cw, cvs in zip(
                     test_data, cv_preds_test, ct_w_test, cv_smooth_test)]
    X_test    = np.array([o[0] for o in test_out])
    R_test    = np.array([o[1] for o in test_out])

    feat_names = make_feature_names()
    log.info(f"피처 수: {X_train.shape[1]}")
    assert X_train.shape[1] == len(feat_names),         f"피처 불일치: {X_train.shape[1]} vs {len(feat_names)}"

    # ── 7. 5-Fold XGBoost 학습 ──────────────────────────────────────────────
    log.info("5-Fold XGBoost 학습 시작 (N_AUG=4)...")
    oof_preds, xgb_test_res, imp_acc = train_group(
        log, "All",
        train_data, X_train, R_train,
        blend_train, true_xyz,
        cv_preds_train, ct_w_train, disp_scale_train,
        X_test, R_test, disp_scale_test,
        feat_names,
        cv_smooth_g=cv_smooth_train,
        cv_smooth_test=cv_smooth_test,
        N_AUG=4,
    )
    log.info(f"\n[v25-OOF]  R-Hit={r_hit(oof_preds, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(oof_preds, true_xyz):.2f}cm")

    # ── 8. 메타 게이팅 ───────────────────────────────────────────────────────
    oof_err   = np.linalg.norm(oof_preds - true_xyz, axis=1)
    blend_err = np.linalg.norm(blend_train - true_xyz, axis=1)
    gate_labels = (oof_err < blend_err).astype(int)
    log.info(f"\n[Gate] 개선={gate_labels.sum()}개  "
             f"악화={(~gate_labels.astype(bool)).sum()}개  "
             f"개선율={gate_labels.mean()*100:.1f}%")

    gate_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        tree_method='hist', random_state=42, n_jobs=-1, verbosity=0,
    )
    gate_clf.fit(X_train, gate_labels)

    gate_prob_test  = gate_clf.predict_proba(X_test)[:, 1]
    gate_prob_train = gate_clf.predict_proba(X_train)[:, 1]

    xgb_train_res = oof_preds - blend_train
    oof_gated     = blend_train + gate_prob_train[:, None] * xgb_train_res
    log.info(f"[OOF-Gated] R-Hit={r_hit(oof_gated, true_xyz):.4f}  "
             f"(raw OOF={r_hit(oof_preds, true_xyz):.4f})")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    run_error_analysis(log, train_data, true_xyz, oof_preds,
                       cv_preds_train, blend_train, ct_w_train,
                       train_ids, out_dir)

    # ── 9. 피처 중요도 ────────────────────────────────────────────────────────
    top_idx = np.argsort(imp_acc)[::-1][:50]
    log.info(f"\n{'='*66}")
    log.info("Top 50 피처 중요도 (5-fold 평균)")
    log.info(f"{'='*66}")
    for rank, i in enumerate(top_idx, 1):
        log.info(f"  {rank:2d}. {feat_names[i]:<42s}  {imp_acc[i]:.4f}")
    pd.DataFrame({'feature': feat_names, 'importance_mean': imp_acc}
                 ).sort_values('importance_mean', ascending=False
                               ).to_csv(out_dir / "feature_importance_v25.csv", index=False)
    log.info("피처 중요도 저장 → output/feature_importance_v25.csv")

    # ── 10. 최종 예측 ────────────────────────────────────────────────────────
    final_test = blend_test + gate_prob_test[:, None] * xgb_test_res

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_test)}
    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )
    out_sub = out_dir / "submission_xgb_v25.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)
    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
