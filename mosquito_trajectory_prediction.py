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


# ── Dataset ────────────────────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    """
    입력: 속도 시계열 (10, 3) — 절대 좌표 대신 상대 운동만 사용
    타깃: CV-last 대비 잔차 (3,)
    """
    def __init__(self, data: list[np.ndarray], cv_preds: np.ndarray,
                 residuals: np.ndarray | None = None):
        self.vels      = np.array([np.diff(t, axis=0) / DT for t in data],
                                  dtype=np.float32)   # (N, 10, 3)
        self.cv_delta  = (cv_preds - np.array([t[-1] for t in data])).astype(np.float32)
        self.residuals = residuals.astype(np.float32) if residuals is not None else None

    def __len__(self):
        return len(self.vels)

    def __getitem__(self, idx):
        x = self.vels[idx]                            # (10, 3)
        cv = self.cv_delta[idx]                       # (3,)
        if self.residuals is not None:
            return x, cv, self.residuals[idx]
        return x, cv


# ── Model ──────────────────────────────────────────────────────────────────────
class MosquitoLSTM(nn.Module):
    """
    속도 시계열 → LSTM → 잔차 예측
    CV-last delta를 FC 입력에 concat → 보정 방향 앵커
    """
    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden + 3, 64),   # +3 for cv_delta
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, vels, cv_delta):
        _, (h, _) = self.lstm(vels)
        h_last = h[-1]                                # (batch, hidden)
        x = torch.cat([h_last, cv_delta], dim=1)     # (batch, hidden+3)
        return self.fc(x)


# ── Metrics ────────────────────────────────────────────────────────────────────
def r_hit(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1) <= 0.01))


def mean_dist_cm(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(preds - trues, axis=1)) * 100)


# ── Train / Eval ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for vels, cv_delta, target in loader:
        vels, cv_delta, target = vels.to(DEVICE), cv_delta.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        pred = model(vels, cv_delta)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(vels)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        vels, cv_delta = batch[0].to(DEVICE), batch[1].to(DEVICE)
        preds.append(model(vels, cv_delta).cpu().numpy())
    return np.concatenate(preds)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log = setup_logger()
    log.info("=" * 50)
    log.info(f"모기 비행 궤적 예측 v4 (LSTM 잔차학습)  device={DEVICE}")
    log.info("=" * 50)

    train_ids, train_data = load_dir(TRAIN_DIR)
    test_ids,  test_data  = load_dir(TEST_DIR)
    log.info(f"Train: {len(train_data)}개  |  Test: {len(test_data)}개")

    labels   = pd.read_csv(LABELS_CSV, index_col='id')
    true_xyz = labels.loc[train_ids, ['x', 'y', 'z']].values

    # CV-last 기준선
    cv_preds_train = batch_cv_last(train_data)
    cv_preds_test  = batch_cv_last(test_data)
    log.info(f"[CV-last]  R-Hit={r_hit(cv_preds_train, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(cv_preds_train, true_xyz):.2f}cm")

    # 잔차 타깃
    residuals_train = (true_xyz - cv_preds_train).astype(np.float32)

    # Train / Val 분리 (80 / 20)
    n = len(train_data)
    val_size = int(n * 0.2)
    idx = np.random.RandomState(42).permutation(n)
    tr_idx, val_idx = idx[val_size:], idx[:val_size]

    tr_data  = [train_data[i] for i in tr_idx]
    val_data = [train_data[i] for i in val_idx]
    tr_cv    = cv_preds_train[tr_idx]
    val_cv   = cv_preds_train[val_idx]
    tr_res   = residuals_train[tr_idx]
    val_res  = residuals_train[val_idx]
    val_true = true_xyz[val_idx]

    tr_ds  = TrajectoryDataset(tr_data,  tr_cv,  tr_res)
    val_ds = TrajectoryDataset(val_data, val_cv, val_res)
    test_ds = TrajectoryDataset(test_data, cv_preds_test)

    tr_loader   = DataLoader(tr_ds,   batch_size=256, shuffle=True,  num_workers=0)
    val_loader  = DataLoader(val_ds,  batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    # 모델
    model = MosquitoLSTM(hidden=128, layers=2, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.HuberLoss(delta=0.005)   # 1cm 스케일에 맞춘 delta

    log.info(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 학습
    best_val_rhit = 0.0
    best_state    = None
    patience      = 30
    no_improve    = 0

    for epoch in range(1, 201):
        tr_loss = train_epoch(model, tr_loader, optimizer, criterion)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            val_res_pred = predict(model, val_loader)
            val_preds    = val_cv + val_res_pred
            val_rhit     = r_hit(val_preds, val_true)
            val_dist     = mean_dist_cm(val_preds, val_true)
            lr_now       = scheduler.get_last_lr()[0]
            log.info(f"Epoch {epoch:3d}  loss={tr_loss:.6f}  "
                     f"val R-Hit={val_rhit:.4f}  val dist={val_dist:.2f}cm  lr={lr_now:.5f}")

            if val_rhit > best_val_rhit:
                best_val_rhit = val_rhit
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 10

            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch}  best val R-Hit={best_val_rhit:.4f}")
                break

    # 최적 모델로 복구
    model.load_state_dict(best_state)

    # Train 전체 성능
    tr_all_ds     = TrajectoryDataset(train_data, cv_preds_train, residuals_train)
    tr_all_loader = DataLoader(tr_all_ds, batch_size=512, shuffle=False, num_workers=0)
    tr_res_pred   = predict(model, tr_all_loader)
    tr_final      = cv_preds_train + tr_res_pred
    log.info(f"[LSTM-train] R-Hit={r_hit(tr_final, true_xyz):.4f}  "
             f"MeanDist={mean_dist_cm(tr_final, true_xyz):.2f}cm  (과적합 참고용)")
    log.info(f"[LSTM-val]   R-Hit={best_val_rhit:.4f}  (best)")

    # 제출
    test_res_pred = predict(model, test_loader)
    final_preds_test = cv_preds_test + test_res_pred

    sub      = pd.read_csv(SAMPLE_SUB)
    pred_map = {tid: pred for tid, pred in zip(test_ids, final_preds_test)}

    for ci, col in enumerate(['x', 'y', 'z']):
        sub[col] = sub['id'].map(
            lambda sid, c=ci: pred_map[sid][c] if sid in pred_map else 0.0
        )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_sub = out_dir / "submission_lstm_v4.csv"
    sub.to_csv(out_sub, index=False)
    os.chmod(out_sub, 0o666)
    os.chmod(out_dir, 0o777)

    log.info(f"제출 파일 저장 → {out_sub}")
    log.info("완료")


if __name__ == '__main__':
    main()
