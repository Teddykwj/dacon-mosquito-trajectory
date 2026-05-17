import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
LABELS_CSV = DATA_DIR / "train_labels.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

DT      = 0.04
HORIZON = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    fh = logging.FileHandler(log_dir / "v6_log.txt", encoding="utf-8")
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
def _turn_cos(vels: np.ndarray) -> np.ndarray:
    """Cosine similarity between consecutive velocity vectors. (10,)"""
    eps = 1e-8
    norms = np.linalg.norm(vels, axis=1)  # (10,)
    cos = np.zeros(len(vels))
    for i in range(1, len(vels)):
        denom = norms[i] * norms[i - 1] + eps
        cos[i] = np.dot(vels[i], vels[i - 1]) / denom
    return cos


def make_seq_features(traj: np.ndarray) -> np.ndarray:
    """
    LSTM 입력: (10, 9) per-timestep 시퀀스
    [vx, vy, vz, ax, ay, az, speed, acc_mag, turn_cos]
    """
    vels  = np.diff(traj, axis=0) / DT            # (10, 3)
    accs  = np.diff(vels, axis=0) / DT             # (9, 3)
    accs  = np.vstack([np.zeros((1, 3)), accs])    # (10, 3)

    speed    = np.linalg.norm(vels, axis=1, keepdims=True)   # (10, 1)
    acc_mag  = np.linalg.norm(accs, axis=1, keepdims=True)   # (10, 1)
    turn_cos = _turn_cos(vels).reshape(-1, 1)                 # (10, 1)

    return np.concatenate([vels, accs, speed, acc_mag, turn_cos], axis=1).astype(np.float32)  # (10, 9)


def make_global_features(traj: np.ndarray, cv_pred: np.ndarray) -> np.ndarray:
    """
    FC에 concat되는 글로벌 피처 (29,)
    """
    vels  = np.diff(traj, axis=0) / DT    # (10, 3)
    accs  = np.diff(vels, axis=0) / DT    # (9, 3)
    speed = np.linalg.norm(vels, axis=1)  # (10,)

    recent_vel  = vels[-3:].mean(axis=0)
    overall_vel = vels.mean(axis=0)

    # turn angle stats (4)
    turn_cos      = _turn_cos(vels)       # (10,)
    turn_mean     = turn_cos[1:].mean()
    turn_std      = turn_cos[1:].std()
    turn_min      = turn_cos[1:].min()
    turn_last     = turn_cos[-1]

    # speed change rate stats (2)
    speed_delta   = np.diff(speed)        # (9,)
    spd_chg_mean  = speed_delta.mean()
    spd_chg_std   = speed_delta.std()

    # last velocity unit vector (3)
    last_vel_norm = np.linalg.norm(vels[-1]) + 1e-8
    last_vel_unit = vels[-1] / last_vel_norm

    # CV-delta alignment with last velocity direction (1)
    cv_delta      = cv_pred - traj[-1]
    cv_delta_norm = np.linalg.norm(cv_delta) + 1e-8
    cv_align      = np.dot(cv_delta / cv_delta_norm, last_vel_unit)

    # speed slope — linear fit over all 10 steps (1)
    t_idx         = np.arange(len(speed))
    speed_slope   = float(np.polyfit(t_idx, speed, 1)[0])

    feats = np.concatenate([
        cv_delta,                                          # CV-last delta (3)
        recent_vel - overall_vel,                          # 최근 방향 전환 세기 (3)
        accs[-3:].mean(axis=0),                            # 최근 가속도 (3)
        accs.std(axis=0),                                  # 가속도 변동성 (3)
        [speed.mean(), speed.std(), speed[-1]],            # 속력 통계 (3)
        traj[-1] - traj[0],                                # 전체 변위 (3)
        [turn_mean, turn_std, turn_min, turn_last],        # 방향 전환 통계 (4)
        [spd_chg_mean, spd_chg_std],                       # 속력 변화율 통계 (2)
        last_vel_unit,                                     # 마지막 속도 단위벡터 (3)
        [cv_align],                                        # CV-delta 정렬도 (1)
        [speed_slope],                                     # 속력 기울기 (1)
    ])
    return feats.astype(np.float32)  # (29,)


# ── Dataset ────────────────────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, data: list[np.ndarray], cv_preds: np.ndarray,
                 residuals: np.ndarray | None = None):
        self.seq     = np.array([make_seq_features(t) for t in data])        # (N, 10, 9)
        self.global_ = np.array([make_global_features(t, cv)
                                  for t, cv in zip(data, cv_preds)])          # (N, 29)
        self.res     = residuals.astype(np.float32) if residuals is not None else None

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if self.res is not None:
            return self.seq[idx], self.global_[idx], self.res[idx]
        return self.seq[idx], self.global_[idx]


# ── Model ──────────────────────────────────────────────────────────────────────
class MosquitoBiLSTM(nn.Module):
    def __init__(self, seq_dim: int = 9, global_dim: int = 29,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        lstm_out = hidden * 2
        self.fc = nn.Sequential(
            nn.Linear(lstm_out + global_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 3),
        )

    def forward(self, seq, global_feat):
        _, (h, _) = self.lstm(seq)
        h_fwd = h[-2]
        h_bwd = h[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)
        x = torch.cat([h_cat, global_feat], dim=1)
        return self.fc(x)


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1) <= 0.01))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Train / Predict ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for seq, glo, target in loader:
        seq, glo, target = seq.to(DEVICE), glo.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(seq, glo), target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(seq)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        seq, glo = batch[0].to(DEVICE), batch[1].to(DEVICE)
        preds.append(model(seq, glo).cpu().numpy())
    return np.concatenate(preds)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 50)
    log.info(f"모기 비행 궤적 예측 v6 (BiLSTM + FE 강화)  device={DEVICE}")
    log.info("=" * 50)

    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)
    log.info(f"Train: {len(train_data)}개  |  Test: {len(test_data)}개")

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    cv_preds_train = batch_cv_last(train_data)
    cv_preds_test  = batch_cv_last(test_data)
    log.info(f"[CV-last]  R-Hit={r_hit(cv_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_preds_train, true_xyz):.2f}cm")

    residuals_train = (true_xyz - cv_preds_train).astype(np.float32)

    # Train / Val 분리
    n = len(train_data)
    idx = np.random.RandomState(42).permutation(n)
    val_n   = int(n * 0.2)
    val_idx = idx[:val_n]
    tr_idx  = idx[val_n:]

    def split(arr, is_list=False):
        if is_list:
            return [arr[i] for i in tr_idx], [arr[i] for i in val_idx]
        return arr[tr_idx], arr[val_idx]

    tr_data,  val_data  = split(train_data, is_list=True)
    tr_cv,    val_cv    = split(cv_preds_train)
    tr_res,   val_res   = split(residuals_train)
    val_true            = true_xyz[val_idx]

    tr_ds   = TrajectoryDataset(tr_data,   tr_cv,  tr_res)
    val_ds  = TrajectoryDataset(val_data,  val_cv, val_res)
    test_ds = TrajectoryDataset(test_data, cv_preds_test)

    tr_loader   = DataLoader(tr_ds,   batch_size=256, shuffle=True,  num_workers=0)
    val_loader  = DataLoader(val_ds,  batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model = MosquitoBiLSTM(seq_dim=9, global_dim=29,
                            hidden=128, layers=2, dropout=0.3).to(DEVICE)
    log.info(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=0.005)

    best_val_rhit = 0.0
    best_state    = None
    no_improve    = 0
    patience      = 30

    for epoch in range(1, 201):
        tr_loss = train_epoch(model, tr_loader, optimizer, criterion)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            val_res_pred = predict(model, val_loader)
            val_preds    = val_cv + val_res_pred
            val_rhit     = r_hit(val_preds, val_true)
            val_dist     = mean_dist_cm(val_preds, val_true)
            log.info(f"Epoch {epoch:3d}  loss={tr_loss:.6f}  "
                     f"val R-Hit={val_rhit:.4f}  val dist={val_dist:.2f}cm")

            if val_rhit > best_val_rhit:
                best_val_rhit = val_rhit
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 10

            if no_improve >= patience:
                log.info(f"Early stopping  epoch={epoch}  best val R-Hit={best_val_rhit:.4f}")
                break

    model.load_state_dict(best_state)

    all_ds     = TrajectoryDataset(train_data, cv_preds_train, residuals_train)
    all_loader = DataLoader(all_ds, batch_size=512, shuffle=False, num_workers=0)
    tr_final   = cv_preds_train + predict(model, all_loader)
    log.info(f"[v6-train] R-Hit={r_hit(tr_final, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(tr_final, true_xyz):.2f}cm  (과적합 참고용)")
    log.info(f"[v6-val]   R-Hit={best_val_rhit:.4f}  (best)")

    # 제출
    final_test = cv_preds_test + predict(model, test_loader)
    sub        = pd.read_csv(SAMPLE_SUB)
    pred_map   = {tid: pred for tid, pred in zip(test_ids, final_test)}

    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_sub = out_dir / "submission_bilstm_v6.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)

    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
