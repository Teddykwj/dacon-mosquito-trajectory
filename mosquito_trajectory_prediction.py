import logging
import os
import sys
import warnings
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

warnings.filterwarnings('ignore', message='.*valid feature names.*')

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
    fh = logging.FileHandler(log_dir / "v39_log.txt", encoding="utf-8")
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
    """등속 선회율(CT) 모델: ω를 마지막 3프레임 평균으로 안정화.
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

    # 법선 가속도 벡터 — 마지막 프레임 기준 (방향)
    a_t_val = float(np.dot(accs_r[-1], v_unit))
    a_n_vec = accs_r[-1] - a_t_val * v_unit
    a_n_mag = float(np.linalg.norm(a_n_vec))

    # [v26] ω 안정화: 마지막 3프레임 ω 평균 (단일 프레임 노이즈 감소)
    omega_vals = []
    for k in range(-3, 0):
        v_k  = vels[k]
        sp_k = float(np.linalg.norm(v_k))
        vu_k = v_k / (sp_k + eps)
        a_k  = accs_r[k]
        at_k = float(np.dot(a_k, vu_k))
        an_k = float(np.linalg.norm(a_k - at_k * vu_k))
        omega_vals.append(an_k / (sp_k + eps))
    omega = float(np.mean(omega_vals))
    theta = omega * dt

    if omega < 1e-6 or speed < 1e-6:
        ct_L = vels[-1] * dt            # 퇴화 → CV와 동일
    else:
        R_curve = speed / (omega + eps)
        n_hat   = a_n_vec / (a_n_mag + eps)
        ct_L = R_curve * np.sin(theta) * v_unit + R_curve * (1 - np.cos(theta)) * n_hat

    ct_pred   = traj[-1] + ct_L @ R    # 로컬 → 글로벌
    ct_weight = float(np.clip(theta / (np.pi / 4), 0.0, 1.0))
    return ct_pred, ct_weight


def _speed_trend_inner(cv_pred: np.ndarray, traj: np.ndarray) -> tuple[np.ndarray, float]:
    """[v26] 접선 가속도 보정 CV: cv_trend = cv + 0.5*a_t*v_unit*dt^2.
    Returns (inner_blend, trend_weight).
    """
    eps  = 1e-8
    dt   = DT * HORIZON
    vg   = np.diff(traj, axis=0) / DT          # (10, 3)
    ag   = np.diff(vg,   axis=0) / DT          # (9, 3)
    spd  = float(np.linalg.norm(vg[-1]))
    vu   = vg[-1] / (spd + eps)
    a_t  = float(np.dot(ag[-1], vu))           # 접선 가속도
    # 접선 방향만 반영 → 원심력 오염 없음
    cv_trend = cv_pred + 0.5 * a_t * vu * dt**2
    # 가중치: |속력 변화율| 기준, 0~0.4 범위
    trend_w  = float(np.clip(abs(a_t) * dt / (spd + eps), 0.0, 0.4))
    inner    = (1.0 - trend_w) * cv_pred + trend_w * cv_trend
    return inner, trend_w


# ── Circle Fitting 앵커 ──────────────────────────────────────────────────────────
def predict_circle_fit(traj: np.ndarray, n_pts: int = 6) -> tuple[np.ndarray, float]:
    """최근 n_pts 점에 원호 피팅 → 다음 위치 예측.
    PCA 투영 후 2D 최소제곱 원 피팅.
    Returns (cf_pred_global, cf_weight).
    cf_weight: 원 피팅 품질 × 선회 각도 기반 [0, 1]
    """
    eps = 1e-9
    dt  = DT * HORIZON

    pts      = traj[-n_pts:].copy()
    last_vel = pts[-1] - pts[-2]
    speed    = float(np.linalg.norm(last_vel))

    if speed < eps:
        return pts[-1] + last_vel, 0.0

    # ── 1. PCA로 궤적 평면 추출 ────────────────────────────────────────────
    center   = pts.mean(axis=0)
    centered = pts - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    u1, u2   = Vt[0], Vt[1]

    # ── 2. 2D 투영 후 최소제곱 원 피팅 ───────────────────────────────────
    p2    = np.stack([centered @ u1, centered @ u2], axis=1)   # (n_pts, 2)
    A     = np.column_stack([2*p2[:, 0], 2*p2[:, 1], np.ones(n_pts)])
    b_vec = p2[:, 0]**2 + p2[:, 1]**2
    res, _, rank, _ = np.linalg.lstsq(A, b_vec, rcond=None)

    if rank < 3:
        return pts[-1] + last_vel, 0.0

    cx, cy = res[0], res[1]
    r2     = cx**2 + cy**2 + res[2]
    if r2 <= 0:
        return pts[-1] + last_vel, 0.0
    r = float(np.sqrt(r2))

    # ── 3. 원호를 따라 dt 전진 ────────────────────────────────────────────
    to_cur      = p2[-1] - np.array([cx, cy])
    to_cur_norm = float(np.linalg.norm(to_cur))
    if to_cur_norm < eps:
        return pts[-1] + last_vel, 0.0

    omega = speed / (r + eps)
    theta = omega * dt

    vel_2d      = p2[-1] - p2[-2]
    tangent_ccw = np.array([-to_cur[1], to_cur[0]]) / to_cur_norm
    sign        = 1.0 if float(np.dot(vel_2d, tangent_ccw)) >= 0 else -1.0
    theta      *= sign

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    new_2d       = np.array([cx, cy]) + np.array([[cos_t, -sin_t], [sin_t, cos_t]]) @ to_cur

    # ── 4. 3D 복원 ────────────────────────────────────────────────────────
    cf_pred = center + new_2d[0] * u1 + new_2d[1] * u2

    # ── 5. 신뢰도: 피팅 잔차 × 선회 강도 ─────────────────────────────────
    radii       = np.sqrt(((p2 - np.array([cx, cy]))**2).sum(axis=1))
    fit_rmse    = float(np.sqrt(((radii - r)**2).mean()))
    fit_quality = float(np.clip(1.0 - fit_rmse / (r + eps), 0.0, 1.0))
    cf_weight   = float(np.clip(abs(theta) * fit_quality, 0.0, 1.0))

    return cf_pred, cf_weight


def batch_physics_blend(data: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """[v26] speed-trend inner + CT 2-layer 블렌드.
    Returns (blend, cv_preds, ct_weights, trend_weights).
    """
    cv_preds   = np.array([predict_cv_last(t) for t in data])
    ct_results = [predict_ct(t) for t in data]
    ct_preds   = np.array([r[0] for r in ct_results])
    ct_weights = np.array([r[1] for r in ct_results])
    trend_res  = [_speed_trend_inner(cv, t) for cv, t in zip(cv_preds, data)]
    inner_preds = np.array([r[0] for r in trend_res])
    trend_ws    = np.array([r[1] for r in trend_res])
    w    = ct_weights[:, None]
    blend = (1 - w) * inner_preds + w * ct_preds
    return blend, cv_preds, ct_weights, trend_ws


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
                       cv_smooth: np.ndarray = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
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

    # ── [NEW v36] cv_smooth 로컬 변위 (불일치도 계산 위해 분리) ──────────────
    if cv_smooth is not None:
        cv_smooth_L         = (cv_smooth - traj[-1]) @ R.T
        cv_smooth_vs_raw_L  = cv_smooth_L - cv_L
        disagree_smooth_ct  = cv_smooth_L - ct_pred_L
        disagree_smooth_ca  = cv_smooth_L - ca_pred_L
    else:
        cv_smooth_L        = np.zeros(3)
        cv_smooth_vs_raw_L = np.zeros(3)
        disagree_smooth_ct = np.zeros(3)
        disagree_smooth_ca = np.zeros(3)
    disagree_ct_ca = ct_pred_L - ca_pred_L

    # ── [NEW v36] 절대 위치 ──────────────────────────────────────────────────
    abs_pos_last  = traj[-1].copy()
    abs_pos_first = traj[0].copy()

    # ── [NEW v36] 3D 비틀림 (Torsion) ────────────────────────────────────────
    torsion = np.zeros(10)
    for _i in range(2, 10):
        _v   = vels[_i];  _vp  = accs[_i];  _vpp = jerk[_i]
        _cr  = np.cross(_v, _vp)
        _csq = float(np.dot(_cr, _cr))
        if _csq > 1e-12:
            torsion[_i] = float(np.dot(_cr, _vpp)) / _csq
    torsion_vals = torsion[2:]   # (8,)

    # ── [NEW v36] ω 변화율 ────────────────────────────────────────────────────
    d_omega       = np.diff(omega)              # (8,)
    d_omega_stats = np.array([float(d_omega[-3:].mean()),
                               float(d_omega.std()),
                               float(d_omega[-1])])

    # ── [NEW v36] 곡률·속력 변동계수 & 원호 피팅 품질 대리 ──────────────────
    kappa_cv        = float(kappa[1:].std()   / (kappa[1:].mean()  + 1e-8))
    speed_cv        = float(speed.std()        / (speed.mean()      + 1e-8))
    kappa_last3_std = float(kappa[-3:].std())

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
        cv_smooth_L,              # (3)
        cv_smooth_vs_raw_L,       # (3)

        # ── [NEW v36] 절대 위치 ──────────────────────────────────────────────
        abs_pos_last,             # (3) 현재 절대 위치
        abs_pos_first,            # (3) 초기 절대 위치

        # ── [NEW v36] 3D 비틀림 ──────────────────────────────────────────────
        torsion_vals,             # (8) 프레임별 비틀림
        [torsion_vals.mean(), torsion_vals.std(),
         torsion[-1], torsion[-3:].mean()],  # (4)

        # ── [NEW v36] 예측 불일치도 ──────────────────────────────────────────
        disagree_smooth_ct,       # (3) cv_smooth vs CT
        disagree_smooth_ca,       # (3) cv_smooth vs CA
        disagree_ct_ca,           # (3) CT vs CA

        # ── [NEW v36] 곡률·속력 변동계수 ─────────────────────────────────────
        [kappa_cv, speed_cv, kappa_last3_std],  # (3)

        # ── [NEW v36] ω 변화율 ────────────────────────────────────────────────
        d_omega,                  # (8)
        d_omega_stats,            # (3)

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

    # [NEW v36] 절대 위치
    for ax in axes: N.append(f"abs_pos_last_{ax}")
    for ax in axes: N.append(f"abs_pos_first_{ax}")

    # [NEW v36] 3D 비틀림
    for t in range(2, 10): N.append(f"torsion_t{t}")
    N += ["torsion_mean", "torsion_std", "torsion_last", "torsion_last3_mean"]

    # [NEW v36] 예측 불일치도
    for ax in axes: N.append(f"disagree_smooth_ct_{ax}")
    for ax in axes: N.append(f"disagree_smooth_ca_{ax}")
    for ax in axes: N.append(f"disagree_ct_ca_{ax}")

    # [NEW v36] 곡률·속력 변동계수
    N += ["kappa_cv", "speed_cv", "kappa_last3_std"]

    # [NEW v36] ω 변화율
    for t in range(1, 9): N.append(f"d_omega_t{t}")
    N += ["d_omega_last3_mean", "d_omega_std", "d_omega_last"]

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
    save_path = out_dir / "oof_analysis_v39.csv"
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
                extra_X=None, extra_y=None,
                seed=42, model_type='xgb', N_AUG=4, n_folds=5):
    """5-Fold 학습 + 3D 회전 증강.
    model_type: 'xgb' | 'lgb'
    extra_X/extra_y: pseudo-label 피처/타겟 (정규화 로컬 잔차).
    Returns (oof, test_res_avg, imp_avg, test_res_per_fold).
    """
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

    kf                = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof               = np.zeros_like(true_xyz_g)
    test_res_acc      = np.zeros((len(disp_scale_test), 3))
    test_res_per_fold = []
    imp_acc           = np.zeros(len(feat_names))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n)), 1):
        tr_aug_idx = np.concatenate([tr_idx + n * r for r in range(N_AUG + 1)])

        if extra_X is not None and len(extra_X) > 0:
            X_fold = np.vstack([X_all[tr_aug_idx], extra_X])
            y_fold = np.vstack([res_loc_norm[tr_aug_idx], extra_y])
        else:
            X_fold = X_all[tr_aug_idx]
            y_fold = res_loc_norm[tr_aug_idx]

        if model_type == 'lgb':
            base = lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
                random_state=seed, n_jobs=-1, verbosity=-1,
            )
        else:
            _xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            base = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                tree_method='hist', device=_xgb_device, random_state=seed, n_jobs=-1, verbosity=0,
            )
        model = MultiOutputRegressor(base, n_jobs=1)
        model.fit(X_fold, y_fold)

        val_res_local  = model.predict(X_all[val_idx]) * disp_scale_g[val_idx, None]
        val_res_global = np.einsum('nji,nj->ni', R_all[val_idx], val_res_local)
        oof[val_idx]   = blend_all_aug[val_idx] + val_res_global
        n_pseudo       = len(extra_X) if extra_X is not None else 0
        log.info(f"  [{label}] Fold {fold}/{n_folds}  "
                 f"R-Hit={r_hit(oof[val_idx], true_xyz_g[val_idx]):.4f}  "
                 f"(n_tr={len(tr_aug_idx):,}+{n_pseudo}pseudo)")

        test_res_local_f  = model.predict(X_test) * disp_scale_test[:, None]
        test_res_global_f = np.einsum('nji,nj->ni', R_test, test_res_local_f)
        test_res_per_fold.append(test_res_global_f)
        test_res_acc     += test_res_global_f
        imp_acc          += np.array([e.feature_importances_
                                      for e in model.estimators_]).mean(0)

    return oof, test_res_acc / n_folds, imp_acc / n_folds, test_res_per_fold


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


# ── DL: GRU 모델 (~68K params) ──────────────────────────────────────────────────
class TrajGRU(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.gru  = nn.GRU(3, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, vel_seq: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(vel_seq)          # h: (1, B, hidden)
        x    = self.drop(h.squeeze(0))   # (B, hidden)
        return self.head(torch.cat([x, phys], dim=-1))


# ── DL: Transformer 모델 (~180K params) ──────────────────────────────────────────
class TrajTransformer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.vel_proj = nn.Linear(3, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, 10, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + 7, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, vel_seq: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        x = self.vel_proj(vel_seq) + self.pos_emb   # (B, 10, d_model)
        x = self.encoder(x)                          # (B, 10, d_model)
        x = x.mean(dim=1)                            # global avg pooling (B, d_model)
        return self.head(torch.cat([x, phys], dim=-1))


# ── DL: GRU 5-Fold 학습 ─────────────────────────────────────────────────────────
def train_gru_5fold(log,
                    train_data, blend_train, true_xyz,
                    ct_results_train, disp_scale_train,
                    test_data, blend_test, ct_results_test, disp_scale_test,
                    N_AUG: int = 19, n_folds: int = 5,
                    n_epochs: int = 100, batch_size: int = 512):
    """5-Fold GRU 학습 + 3D 회전 증강 (속력 기반 정규화, N_AUG=19 → 200K 샘플)."""
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

        model = TrajGRU()
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
        log.info(f"  [GRU] Fold {fold}/{n_folds}  "
                 f"R-Hit={r_hit(oof_preds[val_idx], true_xyz[val_idx]):.4f}  "
                 f"loss={last_loss:.5f}  (n_tr={len(tr_aug_idx):,})")

        with torch.no_grad():
            test_res_norm = model(Vt, Pt).numpy()
        test_res_local  = test_res_norm * disp_scale_test[:, None]
        test_res_global = np.einsum('nji,nj->ni', R_test, test_res_local)
        test_res_acc   += test_res_global

    dl_test_res = test_res_acc / n_folds   # (nt, 3) 평균 잔차 (global)
    return oof_preds, dl_test_res


# ── Smooth R-Hit Loss ─────────────────────────────────────────────────────────
def smooth_rhit_loss(pred_norm: torch.Tensor,
                     true_norm: torch.Tensor,
                     disp_scale: torch.Tensor,
                     k: float = 10.0) -> torch.Tensor:
    """Smooth R-Hit@1cm 손실.
    pred/true_norm: (N, 3) 정규화된 로컬 잔차
    disp_scale: (N,) = speed * dt (미터)
    """
    diff_cm = torch.norm((pred_norm - true_norm) * disp_scale.unsqueeze(1), dim=1) * 100.0
    return -torch.sigmoid(k * (1.0 - diff_cm)).mean()


# ── MLP 모델 (tabular features → 잔차 예측) ──────────────────────────────────
class TrajMLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_feat)
        self.fc1  = nn.Linear(n_feat, hidden)
        self.bn1  = nn.BatchNorm1d(hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.fc3  = nn.Linear(hidden, 256)
        self.bn3  = nn.BatchNorm1d(256)
        self.head = nn.Linear(256, 3)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.input_bn(x)
        h1 = self.drop(self.act(self.bn1(self.fc1(x))))
        h2 = self.drop(self.act(self.bn2(self.fc2(h1)))) + h1  # residual
        h3 = self.drop(self.act(self.bn3(self.fc3(h2))))
        return self.head(h3)


# ── MLP 5-Fold 학습 ───────────────────────────────────────────────────────────
def train_mlp_5fold(log, label,
                    train_data, X_g, R_g,
                    blend_g, true_xyz_g, cv_preds_g, ct_w_g, disp_scale_g,
                    X_test, R_test, disp_scale_test,
                    cv_smooth_g=None,
                    extra_X=None, extra_y=None, extra_scale=None,
                    seed=42, N_AUG=4, n_folds=5,
                    n_epochs=100, batch_size=2048,
                    hidden=512, dropout=0.3,
                    k_rhit=10.0, lr=1e-3, weight_decay=1e-4):
    """5-Fold MLP + smooth R-Hit loss + 3D 회전 증강."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  MLP device: {device}  hidden={hidden}  epochs={n_epochs}")
    n  = len(train_data)
    nt = len(X_test)
    rng = np.random.RandomState(42)

    # ── 증강 데이터 생성 ─────────────────────────────────────────────────────
    all_X = [X_g]; all_R = [R_g]
    all_blend = [blend_g]; all_true = [true_xyz_g]
    all_scale = [disp_scale_g]

    for _ in range(N_AUG):
        Q_batch   = np.array([_random_rotation(rng) for _ in range(n)])
        trajs_rot = [(Q_batch[i] @ train_data[i].T).T for i in range(n)]
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
                      / disp_scale_all[:, None]).astype(np.float32)

    n_feat           = X_all.shape[1]
    kf               = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof              = np.zeros_like(true_xyz_g)
    test_res_acc     = np.zeros((nt, 3))
    test_res_per_fold = []

    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n)), 1):
        tr_aug_idx = np.concatenate([tr_idx + n * r for r in range(N_AUG + 1)])

        X_tr  = X_all[tr_aug_idx].astype(np.float32)
        y_tr  = res_loc_norm[tr_aug_idx]
        s_tr  = disp_scale_all[tr_aug_idx].astype(np.float32)

        if extra_X is not None and len(extra_X) > 0:
            X_tr = np.vstack([X_tr, extra_X.astype(np.float32)])
            y_tr = np.vstack([y_tr, extra_y.astype(np.float32)])
            es   = (extra_scale if extra_scale is not None
                    else np.full(len(extra_X), float(disp_scale_g.mean())))
            s_tr = np.concatenate([s_tr, es.astype(np.float32)])

        X_val = X_all[val_idx].astype(np.float32)
        y_val = res_loc_norm[val_idx]
        s_val = disp_scale_all[val_idx].astype(np.float32)
        R_val = R_all[val_idx]

        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        s_tr_t = torch.from_numpy(s_tr).to(device)
        X_val_t = torch.from_numpy(X_val).to(device)

        torch.manual_seed(seed + fold)
        model = TrajMLP(n_feat, hidden=hidden, dropout=dropout).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        N_tr       = len(X_tr_t)
        best_rhit  = 0.0
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(N_tr, device=device)
            for i in range(0, N_tr, batch_size):
                b    = perm[i:i+batch_size]
                pred = model(X_tr_t[b])
                loss = smooth_rhit_loss(pred, y_tr_t[b], s_tr_t[b], k=k_rhit)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
                model.eval()
                with torch.no_grad():
                    vp = model(X_val_t).cpu().numpy()
                vr = np.einsum('nji,nj->ni', R_val, vp * s_val[:, None])
                vpos = blend_all_aug[val_idx] + vr
                vh   = float(np.mean(np.linalg.norm(vpos - true_xyz_g[val_idx], axis=1) < 0.01))
                if vh > best_rhit:
                    best_rhit  = vh
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_pn  = model(X_val_t).cpu().numpy()
            test_pn = model(X_test_t).cpu().numpy()

        # 극단값 클리핑: 정규화 잔차 ±0.5 (disp_scale × 0.5 ≈ 최대 2cm 보정)
        val_pn  = np.clip(np.nan_to_num(val_pn,  0.0), -0.5, 0.5)
        test_pn = np.clip(np.nan_to_num(test_pn, 0.0), -0.5, 0.5)

        vr_loc = val_pn * s_val[:, None]
        vr_glo = np.einsum('nji,nj->ni', R_val, vr_loc)
        oof[val_idx] = blend_all_aug[val_idx] + vr_glo

        n_pseudo = len(extra_X) if extra_X is not None else 0
        log.info(f"  [{label}] Fold {fold}/{n_folds}  "
                 f"R-Hit={r_hit(oof[val_idx], true_xyz_g[val_idx]):.4f}  "
                 f"(best={best_rhit:.4f}  n_tr={len(tr_aug_idx):,}+{n_pseudo}pseudo)")

        tr_loc = test_pn * disp_scale_test[:, None].astype(np.float32)
        tr_glo = np.einsum('nji,nj->ni', R_test, tr_loc)
        test_res_per_fold.append(tr_glo)
        test_res_acc += tr_glo

    return oof, test_res_acc / n_folds, test_res_per_fold


# ── Transformer 5-Fold 학습 + TTA 추론 ────────────────────────────────────────────
def train_transformer_5fold(log,
                             train_data, blend_train, true_xyz,
                             ct_results_train, disp_scale_train,
                             test_data, blend_test, ct_results_test, disp_scale_test,
                             N_AUG: int = 9, n_folds: int = 5,
                             n_epochs: int = 150, batch_size: int = 2048,
                             n_tta: int = 32, seed: int = 42,
                             k_rhit: float = 10.0):
    """5-Fold TrajTransformer + 3D 회전 증강 + TTA 추론."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Transformer device: {device}  N_AUG={N_AUG}  n_tta={n_tta}  epochs={n_epochs}")
    n   = len(train_data)
    nt  = len(test_data)
    rng = np.random.RandomState(seed)

    def _prep(data, ct_res, disp_sc):
        outs = [prepare_dl_inputs(t, r[0], r[1]) for t, r in zip(data, ct_res)]
        vel  = np.array([o[0] for o in outs])
        phy  = np.array([o[1] for o in outs])
        Rm   = np.array([o[2] for o in outs])
        vel  = (vel * (DT * HORIZON) / disp_sc[:, None, None]).astype(np.float32)
        phy2 = phy.copy();  phy2[:, :6] /= disp_sc[:, None]
        return vel, phy2.astype(np.float32), Rm

    vel_base, phys_base, R_base = _prep(train_data, ct_results_train, disp_scale_train)
    vel_test, phys_test, R_test = _prep(test_data,  ct_results_test,  disp_scale_test)

    res_base = true_xyz - blend_train
    tgt_base = (np.einsum('nij,nj->ni', R_base, res_base)
                / disp_scale_train[:, None]).astype(np.float32)

    all_vel  = [vel_base];  all_phys = [phys_base]
    all_tgt  = [tgt_base];  all_sc   = [disp_scale_train]

    for _ in range(N_AUG):
        Q_batch = np.array([_random_rotation(rng) for _ in range(n)])
        vel_aug = np.zeros_like(vel_base)
        phy_aug = np.zeros_like(phys_base)
        tgt_aug = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            Q  = Q_batch[i]
            tq = (Q @ train_data[i].T).T
            cp = Q @ ct_results_train[i][0]
            vs, ph, Rq = prepare_dl_inputs(tq, cp, ct_results_train[i][1])
            ds = disp_scale_train[i]
            vel_aug[i] = (vs * (DT * HORIZON) / ds).astype(np.float32)
            ph2 = ph.copy();  ph2[:6] /= ds
            phy_aug[i] = ph2.astype(np.float32)
            res_Q = Q @ (true_xyz[i] - blend_train[i])
            tgt_aug[i] = (Rq @ res_Q / ds).astype(np.float32)
        all_vel.append(vel_aug);  all_phys.append(phy_aug)
        all_tgt.append(tgt_aug);  all_sc.append(disp_scale_train)

    vel_all = np.concatenate(all_vel,  axis=0)
    phy_all = np.concatenate(all_phys, axis=0)
    tgt_all = np.concatenate(all_tgt,  axis=0)
    sc_all  = np.concatenate(all_sc,   axis=0).astype(np.float32)

    kf           = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds    = np.zeros_like(true_xyz)
    test_res_acc = np.zeros((nt, 3), dtype=np.float64)
    rng_tta      = np.random.RandomState(seed + 777)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n)), 1):
        tr_aug_idx = np.concatenate([tr_idx + n * r for r in range(N_AUG + 1)])

        X_tr  = torch.from_numpy(vel_all[tr_aug_idx]).to(device)
        P_tr  = torch.from_numpy(phy_all[tr_aug_idx]).to(device)
        Y_tr  = torch.from_numpy(tgt_all[tr_aug_idx]).to(device)
        S_tr  = torch.from_numpy(sc_all[tr_aug_idx]).to(device)
        X_val = torch.from_numpy(vel_base[val_idx]).to(device)
        P_val = torch.from_numpy(phys_base[val_idx]).to(device)

        torch.manual_seed(seed + fold)
        model = TrajTransformer().to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        N_tr      = len(X_tr)
        best_rhit = 0.0
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(N_tr, device=device)
            for i in range(0, N_tr, batch_size):
                b    = perm[i:i+batch_size]
                pred = model(X_tr[b], P_tr[b])
                loss = smooth_rhit_loss(pred, Y_tr[b], S_tr[b], k=k_rhit)
                opt.zero_grad();  loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
                model.eval()
                with torch.no_grad():
                    vp = model(X_val, P_val).cpu().numpy()
                s_v = disp_scale_train[val_idx]
                vr  = np.einsum('nji,nj->ni', R_base[val_idx], vp * s_v[:, None])
                vh  = float(np.mean(np.linalg.norm(
                    blend_train[val_idx] + vr - true_xyz[val_idx], axis=1) < 0.01))
                if vh > best_rhit:
                    best_rhit  = vh
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # OOF (TTA 없이)
        with torch.no_grad():
            val_pn = model(X_val, P_val).cpu().numpy()
        s_v    = disp_scale_train[val_idx]
        vr_gl  = np.einsum('nji,nj->ni', R_base[val_idx], val_pn * s_v[:, None])
        oof_preds[val_idx] = blend_train[val_idx] + vr_gl
        log.info(f"  [TF] Fold {fold}/{n_folds}  "
                 f"R-Hit={r_hit(oof_preds[val_idx], true_xyz[val_idx]):.4f}  "
                 f"(best={best_rhit:.4f}  n_tr={len(tr_aug_idx):,})")

        # Test with TTA: n_tta 회전 후 원래 프레임으로 역변환해 평균
        fold_acc = np.zeros((nt, 3))
        for ti in range(n_tta):
            if ti == 0:
                Vt = torch.from_numpy(vel_test).to(device)
                Pt = torch.from_numpy(phys_test).to(device)
                with torch.no_grad():
                    pr = model(Vt, Pt).cpu().numpy()
                for i in range(nt):
                    fold_acc[i] += R_test[i].T @ (pr[i] * disp_scale_test[i])
            else:
                Q_b  = np.array([_random_rotation(rng_tta) for _ in range(nt)])
                tr_  = [(Q_b[i] @ test_data[i].T).T for i in range(nt)]
                ct_  = [Q_b[i] @ ct_results_test[i][0] for i in range(nt)]
                oo   = [prepare_dl_inputs(tr_[i], ct_[i], ct_results_test[i][1])
                        for i in range(nt)]
                vr_  = np.array([(oo[i][0] * DT * HORIZON / disp_scale_test[i]).astype(np.float32)
                                 for i in range(nt)])
                pr_  = np.array([oo[i][1].copy() for i in range(nt)])
                pr_[:, :6] /= disp_scale_test[:, None]
                Rr_  = np.array([oo[i][2] for i in range(nt)])
                Vr = torch.from_numpy(vr_).to(device)
                Pr = torch.from_numpy(pr_.astype(np.float32)).to(device)
                with torch.no_grad():
                    pr2 = model(Vr, Pr).cpu().numpy()
                for i in range(nt):
                    res_rg = Rr_[i].T @ (pr2[i] * disp_scale_test[i])
                    fold_acc[i] += Q_b[i].T @ res_rg   # Q-frame → original frame

        test_res_acc += fold_acc / n_tta

    return oof_preds, test_res_acc / n_folds


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 66)
    log.info("모기 비행 궤적 예측 v39 (MLP + Transformer TTA 앙상블 + XGB 게이팅)")
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

    # ── 2. CT 결과 (피처용으로만 사용) ──────────────────────────────────────
    ct_results_train = [predict_ct(t) for t in train_data]
    ct_preds_train   = np.array([r[0] for r in ct_results_train])
    ct_w_train       = np.array([r[1] for r in ct_results_train])

    ct_results_test  = [predict_ct(t) for t in test_data]
    ct_preds_test    = np.array([r[0] for r in ct_results_test])
    ct_w_test        = np.array([r[1] for r in ct_results_test])

    log.info(f"[CT (피처용)] weight 평균: {ct_w_train.mean():.3f}")

    # ── 3. Kalman 스무딩 → cv_smooth를 앵커로 승격 ──────────────────────────
    log.info("Kalman 스무딩 계산 중...")
    ka_results_train = [predict_ca_kf(t) for t in train_data]
    cv_smooth_train  = np.array([r[2] for r in ka_results_train])
    ka_results_test  = [predict_ca_kf(t) for t in test_data]
    cv_smooth_test   = np.array([r[2] for r in ka_results_test])

    blend_train = cv_smooth_train
    blend_test  = cv_smooth_test
    log.info(f"[cv_smooth 앵커] R-Hit={r_hit(blend_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(blend_train, true_xyz):.2f}cm")

    # ── 4. 속력 기반 정규화 스케일 ──────────────────────────────────────────
    speed_train      = np.array([np.linalg.norm((t[-1] - t[-2]) / DT) for t in train_data])
    disp_scale_train = np.maximum(speed_train * DT * HORIZON, 0.01)

    speed_test       = np.array([np.linalg.norm((t[-1] - t[-2]) / DT) for t in test_data])
    disp_scale_test  = np.maximum(speed_test  * DT * HORIZON, 0.01)

    # ── 5. 피처 생성 ─────────────────────────────────────────────────────────
    log.info("피처 생성 중...")
    train_out  = [make_xgb_features(t, cv, cw, cvs)
                  for t, cv, cw, cvs in zip(
                      train_data, cv_preds_train, ct_w_train, cv_smooth_train)]
    X_train    = np.array([o[0] for o in train_out])
    R_train    = np.array([o[1] for o in train_out])

    test_out   = [make_xgb_features(t, cv, cw, cvs)
                  for t, cv, cw, cvs in zip(
                      test_data, cv_preds_test, ct_w_test, cv_smooth_test)]
    X_test     = np.array([o[0] for o in test_out])
    R_test     = np.array([o[1] for o in test_out])
    feat_names = make_feature_names()
    log.info(f"피처 수: {X_train.shape[1]}")

    N_PSEUDO = int(0.4 * len(test_data))

    SEEDS   = [42, 123, 456]
    MLP_KW  = dict(
        train_data=train_data, X_g=X_train, R_g=R_train,
        blend_g=blend_train, true_xyz_g=true_xyz,
        cv_preds_g=cv_preds_train, ct_w_g=ct_w_train, disp_scale_g=disp_scale_train,
        X_test=X_test, R_test=R_test, disp_scale_test=disp_scale_test,
        cv_smooth_g=cv_smooth_train,
        N_AUG=9, n_epochs=150, batch_size=2048,
        hidden=512, dropout=0.3, k_rhit=10.0, lr=1e-3,
    )

    # ── 6. Multi-seed MLP 앙상블 ────────────────────────────────────────────
    log.info(f"\n=== Multi-seed MLP 앙상블 (seeds={SEEDS}, N_AUG=9, epochs=150) ===")
    oof_acc      = np.zeros_like(true_xyz, dtype=float)
    test_res_acc = np.zeros((len(test_data), 3))

    for si, seed in enumerate(SEEDS, 1):
        log.info(f"\n--- Seed {seed} ({si}/{len(SEEDS)}) ---")
        oof_s, res_s, _ = train_mlp_5fold(log, f"s{seed}", seed=seed, **MLP_KW)
        oof_acc      += oof_s
        test_res_acc += res_s
        log.info(f"  [Seed {seed}] OOF: {r_hit(oof_s, true_xyz):.4f}")

    oof_preds = oof_acc      / len(SEEDS)
    test_res  = test_res_acc / len(SEEDS)
    log.info(f"\n[v38-Ensemble-OOF]  R-Hit={r_hit(oof_preds, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(oof_preds, true_xyz):.2f}cm")

    # ── 7. Transformer 앙상블 ────────────────────────────────────────────────
    TF_KW = dict(
        train_data=train_data, blend_train=blend_train, true_xyz=true_xyz,
        ct_results_train=ct_results_train, disp_scale_train=disp_scale_train,
        test_data=test_data, blend_test=blend_test,
        ct_results_test=ct_results_test, disp_scale_test=disp_scale_test,
        N_AUG=9, n_epochs=150, batch_size=2048, n_tta=32, k_rhit=10.0,
    )
    log.info(f"\n=== Transformer 앙상블 (seeds={SEEDS}, N_AUG=9, TTA=32) ===")
    tf_oof_acc  = np.zeros_like(true_xyz, dtype=float)
    tf_test_acc = np.zeros((len(test_data), 3))
    for si, seed in enumerate(SEEDS, 1):
        log.info(f"\n--- [TF] Seed {seed} ({si}/{len(SEEDS)}) ---")
        tf_oof_s, tf_res_s = train_transformer_5fold(log, seed=seed, **TF_KW)
        tf_oof_acc  += tf_oof_s
        tf_test_acc += tf_res_s
        log.info(f"  [TF Seed {seed}] OOF: {r_hit(tf_oof_s, true_xyz):.4f}")
    tf_oof      = tf_oof_acc  / len(SEEDS)
    tf_test_res = tf_test_acc / len(SEEDS)
    log.info(f"\n[TF-Ensemble-OOF]  R-Hit={r_hit(tf_oof, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(tf_oof, true_xyz):.2f}cm")

    # ── 8. MLP + Transformer 가중 블렌딩 ────────────────────────────────────
    mlp_rhit = r_hit(oof_preds, true_xyz)
    tf_rhit  = r_hit(tf_oof,    true_xyz)
    w_mlp    = mlp_rhit / (mlp_rhit + tf_rhit)
    w_tf     = 1.0 - w_mlp
    log.info(f"\n[앙상블 가중치]  MLP={w_mlp:.3f}  TF={w_tf:.3f}")
    ens_oof      = w_mlp * oof_preds + w_tf * tf_oof
    ens_test_res = w_mlp * test_res  + w_tf * tf_test_res
    log.info(f"[Ensemble-OOF]  R-Hit={r_hit(ens_oof, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(ens_oof, true_xyz):.2f}cm")

    # ── 9. XGBoost 메타 게이팅 ───────────────────────────────────────────────
    blend_err   = np.linalg.norm(blend_train - true_xyz, axis=1)
    ens_err     = np.linalg.norm(ens_oof     - true_xyz, axis=1)
    gate_labels = (ens_err < blend_err).astype(int)
    log.info(f"\n[Gate] 개선={gate_labels.sum()}개  "
             f"악화={(~gate_labels.astype(bool)).sum()}개  "
             f"개선율={gate_labels.mean()*100:.1f}%")

    _xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        tree_method='hist', device=_xgb_device, random_state=42, n_jobs=-1, verbosity=0,
    )
    gate_clf.fit(X_train, gate_labels)
    gate_prob_test  = gate_clf.predict_proba(X_test)[:, 1]
    gate_prob_train = gate_clf.predict_proba(X_train)[:, 1]

    ens_res_train = ens_oof - blend_train
    oof_gated = blend_train + gate_prob_train[:, None] * ens_res_train
    log.info(f"[OOF-Gated] R-Hit={r_hit(oof_gated, true_xyz):.4f}  "
             f"(raw Ens={r_hit(ens_oof, true_xyz):.4f})")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    run_error_analysis(log, train_data, true_xyz, ens_oof,
                       cv_preds_train, blend_train, ct_w_train,
                       train_ids, out_dir)

    # ── 10. 최종 예측 ────────────────────────────────────────────────────────
    final_test = blend_test + gate_prob_test[:, None] * ens_test_res

    # NaN/Inf 체크 → blend로 폴백
    bad_mask = ~np.isfinite(final_test).all(axis=1)
    if bad_mask.any():
        log.info(f"  경고: {bad_mask.sum()}개 샘플 NaN/Inf → blend_test로 대체")
        final_test[bad_mask] = blend_test[bad_mask]

    # 좌표 범위 검증: blend_test 기준 ±10cm 이내로 클리핑
    MAX_CORR = 0.10
    final_test = np.clip(final_test,
                         blend_test - MAX_CORR,
                         blend_test + MAX_CORR).astype(np.float64)
    log.info(f"[최종 예측] 좌표 범위 검증 완료  "
             f"(blend 기준 ±{MAX_CORR*100:.0f}cm 이내)")

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_test)}
    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )
    out_sub = out_dir / "submission_mlp_v39.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)
    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
