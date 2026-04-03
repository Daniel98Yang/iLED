# gridsearch_koopman.py
#
# Grid search over key hyperparameters of the two-scale Koopman model.
#
# Sweep axes (total combinations printed at startup):
#   important_channel_weight : weight applied to boost channels in recon loss
#   lr_k                     : learning rate for Koopman K and B matrices
#   lr_time_ae               : learning rate for TimeAutoEncoder + memory kernel
#   w_latent                 : weight on latent-prediction MSE in cycle loss
#   w_recon                  : weight on reconstruction MSE in cycle loss
#   stability_coef           : coefficient on the stability eigenvalue penalty
#   joint_recon_coef         : coefficient on recon loss in the joint phase
#
# Resume behaviour:
#   Results are appended line-by-line to RESULTS_FILE (JSONL).
#   On restart the script reads that file, collects completed run_ids,
#   and skips them — so you can kill/restart freely.
#
# Runtime budget:
#   ~10 min per 500-epoch run on your hardware.
#   Combinations are shuffled so partial runs give good coverage.
#   With 3 days ≈ 432 runs you will cover a large fraction of the grid.

import os
import json
import time
import random
import itertools
import traceback
import hashlib
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import joblib

from myautoencoder   import MyAutoEncoder
from smallscaleae    import TimeAutoEncoder
from koopmandataset  import CycleDataset, SequenceDataset
from koopmandynamics import KoopmanDynamics
from sklearn.preprocessing import StandardScaler
from datasets_utils import transform_ts_data

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ PATHS  (unchanged from mytrainer3)
DATA_PATH        = "/content/drive/MyDrive/helicopter_data/class1_train.npy"
CONTROL_PATH     = "/content/drive/MyDrive/helicopter_data/control_class1_train.npz"
VAL_DATA_PATH    = "/content/drive/MyDrive/helicopter_data/class1_test.npy"
VAL_CONTROL_PATH = "/content/drive/MyDrive/helicopter_data/control_class1_test.npz"
AE_PATH          = "/content/iLED/iled/prototsnetresult2/autoencoder_pretrained.pth"
SCALER_PATH      = "/content/iLED/iled/prototsnetresult2/scalers/s2_pf8_pc2_pl0.5_cl0.06_sp-0.03_scaler.pkl"
SAVE_DIR         = "/content/drive/MyDrive/helicopter_data/gridsearch_runs"
RESULTS_FILE     = os.path.join(SAVE_DIR, "results.jsonl")
CTRL_SCALER_SAVE_PATH = "/content/control_scaler.pkl"
TIME_SCALER_SAVE_PATH = "/content/time_scaler.pkl"

# ★ FIXED ARCHITECTURE / TRAINING CONSTANTS
NUM_FEATURES     = 314
SEQ_LEN          = 200
LATENT_DIM       = 8
TIME_LATENT_DIM  = 6
MEMORY_LEN       = 5
MEMORY_HIDDEN    = 32
MEMORY_KERNEL_SZ = 3
BATCH_SIZE_CYCLE = 32
BATCH_SIZE_TIME  = 16
N_EPOCHS         = 500
PRETRAIN_EPOCHS  = 40
KOOPMAN_EPOCHS   = 160
JOINT_EPOCHS     = 300
LR_ALPHA         = 1e-2   # kept fixed across all runs
FREEZE_WINDOW_AE = True

IMPORTANT_CHANNELS = [5,6,7,8,18,19,23,24,25,26,27,28,29,30,31,32,33,34,35,292,293,294,295,296]

# Maximum number of runs to execute (randomly sampled from the full grid).
# The script stops as soon as this many runs have completed successfully,
# resuming previously-completed runs toward this count on restart.
MAX_RUNS = 250

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ GRID DEFINITION
#
#   ~10 min per run · 72 hours ≈ 432 runs budget
#   Full grid size is printed at startup; combinations are shuffled
#   so you get broad coverage even when stopped early.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRID = {
    # boost factor applied to important_channels in weighted recon loss
    "important_channel_weight": [5.0, 15.0, 30.0, 80.0],

    # Koopman K / B learning rate
    "lr_k":                     [1e-3, 3e-3, 1e-2],

    # TimeAutoEncoder + memory kernel learning rate
    "lr_time_ae":               [1e-3, 5e-3, 1e-2],

    # weights inside koopman_loss_cycle
    "w_latent":                 [1.0, 1.5, 2.0],
    "w_recon":                  [0.5, 1.0, 2.0],

    # coefficient on stability eigenvalue penalty  (applied in koopman + joint phases)
    "stability_coef":           [1e-4, 5e-4, 2e-3],

    # coefficient on reconstruction loss in the joint phase
    "joint_recon_coef":         [1e-4, 1e-3, 1e-2],
}


# ─────────────────────────────────────────────────────────
# iLED-style convolutional memory kernel  (identical to mytrainer3)
# ─────────────────────────────────────────────────────────
class ConvMemoryKernel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 32, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size, padding=pad),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        x = z_hist.permute(0, 2, 1)
        x = self.net(x)
        return x.mean(dim=-1)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def load_npz_or_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[list(arr.keys())[0]]
    return arr


def cfg_to_id(cfg: dict) -> str:
    """Stable short hash for a config dict — used as the run directory name."""
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:10]


def load_completed_ids(results_file: str) -> set:
    """Read JSONL results file and return the set of completed run_ids."""
    ids = set()
    if not os.path.exists(results_file):
        return ids
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["run_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def append_result(results_file: str, record: dict):
    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────
# Data loading  (done once, shared across all runs)
# ─────────────────────────────────────────────────────────
def load_all_data():
    sensor_train = load_npz_or_npy(DATA_PATH)
    sensor_val   = load_npz_or_npy(VAL_DATA_PATH)
    ctrl_train   = load_npz_or_npy(CONTROL_PATH)
    ctrl_val     = load_npz_or_npy(VAL_CONTROL_PATH)

    CONTROL_DIM = ctrl_train.shape[-1]

    if os.path.exists(CTRL_SCALER_SAVE_PATH):
        ctrl_scaler = joblib.load(CTRL_SCALER_SAVE_PATH)
    else:
        ctrl_scaler = StandardScaler()
        ctrl_scaler.fit(ctrl_train.reshape(-1, CONTROL_DIM))
        joblib.dump(ctrl_scaler, CTRL_SCALER_SAVE_PATH)

    if os.path.exists(TIME_SCALER_SAVE_PATH):
        time_scaler = joblib.load(TIME_SCALER_SAVE_PATH)
    else:
        data_for_scaler = sensor_train.transpose(0, 2, 1).reshape(-1, NUM_FEATURES)
        time_scaler = StandardScaler()
        time_scaler.fit(data_for_scaler)
        joblib.dump(time_scaler, TIME_SCALER_SAVE_PATH)

    cycle_sklearn_scaler = joblib.load(SCALER_PATH)

    assert sensor_train.shape[1] == NUM_FEATURES
    assert sensor_train.shape[2] == SEQ_LEN

    data = dict(
        sensor_train=sensor_train, sensor_val=sensor_val,
        ctrl_train=ctrl_train, ctrl_val=ctrl_val,
        ctrl_scaler=ctrl_scaler, time_scaler=time_scaler,
        cycle_sklearn_scaler=cycle_sklearn_scaler,
        CONTROL_DIM=CONTROL_DIM,
    )
    print(f"Data loaded — train: {sensor_train.shape}  val: {sensor_val.shape}")
    return data


# ─────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────
def train_one(cfg: dict, run_id: str, shared_data: dict, device: torch.device) -> dict:
    """
    Train one 500-epoch run with the given hyperparameter config.
    Returns a result dict with the best validation losses.
    """
    # ── Unpack config ────────────────────────────────────
    icw          = cfg["important_channel_weight"]
    lr_k         = cfg["lr_k"]
    lr_time_ae   = cfg["lr_time_ae"]
    w_latent     = cfg["w_latent"]
    w_recon      = cfg["w_recon"]
    stab_coef    = cfg["stability_coef"]
    jrecon_coef  = cfg["joint_recon_coef"]

    # ── Unpack shared data ───────────────────────────────
    sensor_train         = shared_data["sensor_train"]
    sensor_val           = shared_data["sensor_val"]
    ctrl_train           = shared_data["ctrl_train"]
    ctrl_val             = shared_data["ctrl_val"]
    ctrl_scaler          = shared_data["ctrl_scaler"]
    time_scaler          = shared_data["time_scaler"]
    cycle_sklearn_scaler = shared_data["cycle_sklearn_scaler"]
    CONTROL_DIM          = shared_data["CONTROL_DIM"]

    run_dir = os.path.join(SAVE_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ── Datasets & loaders ──────────────────────────────
    cycle_train_ds = CycleDataset(sensor_train, ctrl_train)
    cycle_val_ds   = CycleDataset(sensor_val,   ctrl_val)
    time_train_ds  = SequenceDataset(sensor_train, controls=ctrl_train)
    time_val_ds    = SequenceDataset(sensor_val,   controls=ctrl_val)

    cycle_train_loader = DataLoader(cycle_train_ds, batch_size=BATCH_SIZE_CYCLE,
                                    shuffle=True,  num_workers=2)
    cycle_val_loader   = DataLoader(cycle_val_ds,   batch_size=BATCH_SIZE_CYCLE,
                                    shuffle=False, num_workers=2)
    time_train_loader  = DataLoader(time_train_ds,  batch_size=BATCH_SIZE_TIME,
                                    shuffle=True,  num_workers=2)
    time_val_loader    = DataLoader(time_val_ds,    batch_size=BATCH_SIZE_TIME,
                                    shuffle=False, num_workers=2)

    # ── Scaler tensors ───────────────────────────────────
    _mean  = torch.tensor(time_scaler.mean_,  dtype=torch.float32).to(device)
    _scale = torch.tensor(time_scaler.scale_, dtype=torch.float32).to(device)
    mean_ts  = _mean.view(1, -1)
    scale_ts = _scale.view(1, -1)

    _meanc  = torch.tensor(cycle_sklearn_scaler.mean_,  dtype=torch.float32).to(device)
    _scalec = torch.tensor(cycle_sklearn_scaler.scale_, dtype=torch.float32).to(device)
    mean_cyc  = _meanc.view(1, -1, 1)
    scale_cyc = _scalec.view(1, -1, 1)

    def normalize_cycle_ae(x):
        x_np = x.detach().cpu().numpy()
        x_scaled = np.empty_like(x_np)
        for i in range(x_np.shape[0]):
            x_scaled[i] = cycle_sklearn_scaler.transform(x_np[i].T).T
        return torch.tensor(x_scaled, dtype=torch.float32).to(device)

    def denormalize_cycle_ae(x):
        x_np = x.detach().cpu().numpy()
        x_inv = np.empty_like(x_np)
        for i in range(x_np.shape[0]):
            x_inv[i] = cycle_sklearn_scaler.inverse_transform(x_np[i].T).T
        return torch.tensor(x_inv, dtype=torch.float32).to(device)

    def normalize_time(x):
        return (x - mean_ts) / (scale_ts + 1e-8)

    def normalize_control(u):
        u_np = u.detach().cpu().numpy()
        u_scaled = ctrl_scaler.transform(u_np).astype(np.float32)
        return torch.tensor(u_scaled).to(device)

    # ── Models ───────────────────────────────────────────
    cycle_ae = MyAutoEncoder(
        num_features=NUM_FEATURES, latent_features=LATENT_DIM, seq_len=SEQ_LEN,
    ).to(device)
    state_dict = torch.load(AE_PATH, map_location=device)
    new_state_dict = {"model." + k: v for k, v in state_dict.items()}
    cycle_ae.load_state_dict(new_state_dict)
    if FREEZE_WINDOW_AE:
        for p in cycle_ae.parameters():
            p.requires_grad = False
        cycle_ae.eval()

    time_ae       = TimeAutoEncoder(input_dim=NUM_FEATURES, latent_dim=TIME_LATENT_DIM).to(device)
    cycle_dynamics = KoopmanDynamics(latent_dim=LATENT_DIM,      control_dim=CONTROL_DIM).to(device)
    time_dynamics  = KoopmanDynamics(latent_dim=TIME_LATENT_DIM, control_dim=CONTROL_DIM).to(device)
    memory_kernel  = ConvMemoryKernel(
        latent_dim=TIME_LATENT_DIM, hidden_dim=MEMORY_HIDDEN, kernel_size=MEMORY_KERNEL_SZ,
    ).to(device)
    alpha = nn.Parameter(torch.tensor(0.0, device=device))

    # ── Optimizer ────────────────────────────────────────
    optimizer = torch.optim.Adam([
        {'params': cycle_dynamics.parameters(), 'lr': lr_k},
        {'params': time_dynamics.parameters(),  'lr': lr_k},
        {'params': time_ae.parameters(),        'lr': lr_time_ae},
        {'params': memory_kernel.parameters(),  'lr': lr_time_ae},
        {'params': cycle_ae.parameters(),       'lr': 1e-4},
        {'params': [alpha],                     'lr': LR_ALPHA},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    k_params  = [cycle_dynamics.K, time_dynamics.K]
    b_params  = [p for p in [cycle_dynamics.B, time_dynamics.B] if p is not None]
    ae_params = list(time_ae.parameters()) + list(memory_kernel.parameters()) + [alpha]

    # ── Weighted recon loss channel mask ─────────────────
    channel_weights = torch.ones(NUM_FEATURES, device=device)
    channel_weights[IMPORTANT_CHANNELS] = icw
    channel_weights_cyc = channel_weights.view(1, -1, 1)  # (1, C, 1)

    # ── Forward passes ───────────────────────────────────
    def forward_cycle(batch):
        x_t    = batch['x_t'].to(device)
        x_next = batch['x_next'].to(device)
        u_t    = batch['u_t'].to(device)
        x_t_norm    = normalize_cycle_ae(x_t)
        x_next_norm = normalize_cycle_ae(x_next)
        with torch.set_grad_enabled(True):
            z           = cycle_ae.encode(x_t_norm)
            z_next      = cycle_ae.encode(x_next_norm)
            recon_norm  = cycle_ae.decode(z)
            z_next_pred = cycle_dynamics(z, u_t)
            recon_pred_norm = cycle_ae.decode(z_next_pred)
        xs = x_t_norm
        recon      = denormalize_cycle_ae(recon_norm)
        recon_pred = denormalize_cycle_ae(recon_pred_norm)
        return {
            'z': z, 'z_next': z_next, 'z_next_pred': z_next_pred,
            'recon': recon, 'recon_pred': recon_pred,
            'recon_norm': recon_norm, 'xs': xs,
            'x_t': x_t, 'x_next': x_next,
        }

    def forward_time(batch):
        x_seq = batch['x_seq'].to(device)
        u_seq = batch['u_seq'].to(device)
        B, T, D = x_seq.shape
        xs       = normalize_time(x_seq.view(B * T, D)).view(B, T, D)
        z_seq    = time_ae.encode(xs.view(B * T, D)).view(B, T, -1)
        xs_recon = time_ae.decode(z_seq.view(B * T, -1)).view(B, T, D)
        u_norm   = normalize_control(u_seq.view(B * T, CONTROL_DIM)).view(B, T, CONTROL_DIM)
        alpha_pos = torch.nn.functional.softplus(alpha)
        preds, targets = [], []
        z_lin_last = z_mem_last = None
        for t in range(MEMORY_LEN, T):
            z_t   = z_seq[:, t - 1]
            u_t   = u_norm[:, t - 1]
            z_lin = time_dynamics(z_t, u_t)
            z_hist = z_seq[:, t - MEMORY_LEN : t]
            z_mem  = memory_kernel(z_hist)
            z_next_pred = z_lin + alpha_pos * z_mem
            preds.append(z_next_pred)
            targets.append(z_seq[:, t])
            z_lin_last, z_mem_last = z_lin, z_mem
        preds   = torch.stack(preds,   dim=1)
        targets = torch.stack(targets, dim=1)
        return {
            'preds': preds, 'targets': targets,
            'xs': xs, 'xs_recon': xs_recon,
            'x_t': x_seq, 'z_seq': z_seq,
            'z_lin_last': z_lin_last, 'z_mem_last': z_mem_last,
        }

    def koopman_loss_cycle(out):
        loss_latent = ((out['z_next_pred'] - out['z_next']) ** 2).mean()
        loss_recon  = ((out['recon_norm']  - out['xs'])     ** 2 * channel_weights_cyc).mean()
        return w_latent * loss_latent + w_recon * loss_recon

    def stability_penalty(K):
        return torch.relu(torch.linalg.eigvals(K).abs() - 1.0).mean()

    # ── Training loop ────────────────────────────────────
    best_val_loss = float('inf')
    best_val_cyc  = float('inf')
    best_val_ts   = float('inf')

    for epoch in range(1, N_EPOCHS + 1):
        if epoch <= PRETRAIN_EPOCHS:
            phase = "pretrain"
        elif epoch <= PRETRAIN_EPOCHS + KOOPMAN_EPOCHS:
            phase = "koopman"
        else:
            phase = "joint"

        # requires_grad gating
        if phase == "pretrain":
            time_ae.train()
            cycle_ae.eval()
            for p in time_ae.parameters():        p.requires_grad = True
            for p in cycle_ae.parameters():       p.requires_grad = False
            for p in cycle_dynamics.parameters(): p.requires_grad = False
            for p in time_dynamics.parameters():  p.requires_grad = False
            for p in memory_kernel.parameters():  p.requires_grad = False
            alpha.requires_grad = False
        elif phase == "koopman":
            time_ae.eval();  cycle_ae.eval()
            for p in time_ae.parameters():        p.requires_grad = False
            for p in cycle_ae.parameters():       p.requires_grad = False
            for p in cycle_dynamics.parameters(): p.requires_grad = True
            for p in time_dynamics.parameters():  p.requires_grad = True
            for p in memory_kernel.parameters():  p.requires_grad = True
            alpha.requires_grad = True
        else:  # joint
            time_ae.train();  cycle_ae.train()
            for p in time_ae.parameters():        p.requires_grad = True
            for p in cycle_ae.parameters():       p.requires_grad = True
            for p in cycle_dynamics.parameters(): p.requires_grad = True
            for p in time_dynamics.parameters():  p.requires_grad = True
            for p in memory_kernel.parameters():  p.requires_grad = True
            alpha.requires_grad = True

        cycle_dynamics.train()
        time_dynamics.train()
        memory_kernel.train()
        cycle_ae.eval()   # keep BN stats fixed

        tr_cyc, tr_ts = [], []
        time_iter = iter(time_train_loader)

        for cyc_batch in cycle_train_loader:
            try:
                ts_batch = next(time_iter)
            except StopIteration:
                time_iter = iter(time_train_loader)
                ts_batch  = next(time_iter)

            optimizer.zero_grad()

            out_cyc       = forward_cycle(cyc_batch)
            loss_cyc      = koopman_loss_cycle(out_cyc)
            out_ts        = forward_time(ts_batch)
            loss_latent   = ((out_ts['preds'] - out_ts['targets']) ** 2).mean()
            loss_recon_ts = ((out_ts['xs_recon'] - out_ts['xs']) ** 2).mean()

            if phase == "pretrain":
                loss = loss_recon_ts
            elif phase == "koopman":
                loss = loss_cyc + loss_latent
            else:  # joint
                loss = loss_cyc + loss_latent + jrecon_coef * loss_recon_ts

            if phase != "pretrain":
                stab = stability_penalty(cycle_dynamics.K) + stability_penalty(time_dynamics.K)
                loss = loss + stab_coef * stab

            loss.backward()
            torch.nn.utils.clip_grad_norm_(k_params,  max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(b_params,  max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(ae_params, max_norm=1.0)
            optimizer.step()

            if phase == "pretrain":
                tr_cyc.append(0.0)
                tr_ts.append(loss_recon_ts.detach().item())
            else:
                tr_cyc.append(loss_cyc.detach().item())
                tr_ts.append(loss_latent.detach().item())

        # ── Validation ─────────────────────────────────
        cycle_dynamics.eval(); time_dynamics.eval()
        time_ae.eval(); memory_kernel.eval(); cycle_ae.eval()

        va_cyc, va_ts = [], []
        with torch.no_grad():
            for batch in cycle_val_loader:
                va_cyc.append(koopman_loss_cycle(forward_cycle(batch)).item())
            for batch in time_val_loader:
                out_t = forward_time(batch)
                va_ts.append(((out_t['preds'] - out_t['targets'])**2).mean().item())

        va_c = float(np.mean(va_cyc))
        va_t = float(np.mean(va_ts))
        va_total = va_c + va_t

        scheduler.step(va_total)

        if va_total < best_val_loss:
            best_val_loss = va_total
            best_val_cyc  = va_c
            best_val_ts   = va_t
            torch.save({
                'epoch':          epoch,
                'cycle_dynamics': cycle_dynamics.state_dict(),
                'time_dynamics':  time_dynamics.state_dict(),
                'time_ae':        time_ae.state_dict(),
                'memory_kernel':  memory_kernel.state_dict(),
                'alpha':          alpha.data,
                'optimizer':      optimizer.state_dict(),
                'val_loss':       best_val_loss,
                'cfg':            cfg,
            }, os.path.join(run_dir, "best.pth"))

        # Light per-epoch print (no per-batch debug to keep logs clean)
        if epoch % 50 == 0 or epoch == N_EPOCHS:
            print(
                f"  [{run_id}] epoch {epoch:4d}/{N_EPOCHS} | phase={phase:8s} | "
                f"cyc va {va_c:.3e} | ts va {va_t:.3e} | total {va_total:.3e}"
                + (" ← best" if va_total == best_val_loss else "")
            )

    return {
        "run_id":         run_id,
        "cfg":            cfg,
        "best_val_total": best_val_loss,
        "best_val_cyc":   best_val_cyc,
        "best_val_ts":    best_val_ts,
        "best_alpha_softplus": float(torch.nn.functional.softplus(alpha).item()),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main: enumerate, shuffle, run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build all combinations
    keys   = list(GRID.keys())
    values = list(GRID.values())
    all_cfgs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Deterministic shuffle (seed=42) → same order every restart → easy to track progress
    random.seed(42)
    random.shuffle(all_cfgs)

    total = len(all_cfgs)
    print(f"\nGrid size: {total} combinations")
    print(f"Estimated runtime: {total * 10 / 60:.1f} hours  "
          f"({total * 10:.0f} min @ ~10 min/run)\n")

    # Load completed run_ids so we can skip them on restart
    completed = load_completed_ids(RESULTS_FILE)
    print(f"Already completed: {len(completed)} / {total}")

    # Load data once (shared across all runs)
    shared_data = load_all_data()

    run_number = len(completed)

    for cfg in all_cfgs:
        if len(completed) >= MAX_RUNS:
            break

        run_id = cfg_to_id(cfg)

        if run_id in completed:
            continue

        run_number += 1
        print(f"\n{'='*70}")
        print(f"Run {run_number} / {MAX_RUNS}   id={run_id}")
        print(f"  Config: {json.dumps(cfg, indent=4)}")

        t0 = time.time()
        try:
            result = train_one(cfg, run_id, shared_data, device)
            elapsed = time.time() - t0
            result["elapsed_seconds"] = elapsed

            append_result(RESULTS_FILE, result)
            completed.add(run_id)

            print(f"  ✅  best val total={result['best_val_total']:.4e}  "
                  f"cyc={result['best_val_cyc']:.4e}  ts={result['best_val_ts']:.4e}  "
                  f"({elapsed/60:.1f} min)")

        except Exception:
            elapsed = time.time() - t0
            err_msg = traceback.format_exc()
            print(f"  ❌  FAILED after {elapsed:.0f}s:\n{err_msg}")
            # Log failure so we don't retry in the same session
            append_result(RESULTS_FILE, {
                "run_id": run_id,
                "cfg": cfg,
                "error": err_msg,
                "elapsed_seconds": elapsed,
            })
            completed.add(run_id)

    print(f"\n{'='*70}")
    print(f"Grid search complete.  {len(completed)} / {MAX_RUNS} runs finished.")
    print(f"Results in: {RESULTS_FILE}")

    # ── Quick leaderboard ─────────────────────────────────
    print("\n── Top 10 runs by best_val_total ──")
    records = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "best_val_total" in r:
                    records.append(r)

    records.sort(key=lambda r: r["best_val_total"])
    for rank, r in enumerate(records[:10], 1):
        print(f"  #{rank:2d}  val={r['best_val_total']:.4e}  "
              f"cyc={r['best_val_cyc']:.4e}  ts={r['best_val_ts']:.4e}  "
              f"run_id={r['run_id']}")
        print(f"       {r['cfg']}")


if __name__ == "__main__":
    main()