# mytrainer3.py
#
# Two-scale Koopman training with iLED-style memory:
#   CYCLE scale — pretrained CNN AE (frozen), adjacent window pairs,   latent dim = 8
#                 Pure linear Koopman — no memory (cycle scale untouched)
#   TIME  scale — trainable MLP AE,           full trajectories,       latent dim = 6
#                 Linear Koopman + convolutional Mori-Zwanzig memory kernel (iLED)
#
# Memory design (time scale only):
#   z_{t+1} = K z_t + B u_t + alpha * memory_kernel(z_{t-L:t-1})
#   - memory_kernel: causal 1-D CNN over the latent history window
#   - alpha:         learnable nn.Parameter (Markovian init at 0.0)
#   - kernel weights: zero-initialized so training starts Markovian
#
# Bugs fixed vs backup:
#   1. SequenceDataset replaces KoopmanDataset for the time scale —
#      forward_time now receives 'x_seq'/'u_seq' as expected.
#   2. cycle_dynamics added to the optimizer (was silently never updated).
#   3. Validation loop no longer calls koopman_loss() on time output
#      (wrong keys) and no longer reads missing 'recon'/'x_t' keys.
#   4. Debug block for linear/memory norm no longer references the
#      out-of-scope variables T and 'u_seq' key.
#   5. alpha is now an nn.Parameter, not a plain float.
#   6. memory_net replaced with ConvMemoryKernel (1-D CNN, iLED-style).
#   7. All controls are normalized in a single batched call before the
#      time-step loop (avoids N_timesteps × CPU-GPU round-trips).
#   8. BATCH_SIZE_TIME reduced to match full-trajectory samples.
#   9. k_params / b_params are lists (not generator expressions) so
#      clip_grad_norm_ can iterate them more than once.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import joblib
import os

from myautoencoder   import MyAutoEncoder
from smallscaleae    import TimeAutoEncoder
from koopmandataset  import CycleDataset, SequenceDataset
from koopmandynamics import KoopmanDynamics
from sklearn.preprocessing import StandardScaler
from datasets_utils import transform_ts_data

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ PATHS
DATA_PATH        = "/content/drive/MyDrive/helicopter_data/1class0_train_X.npy"
CONTROL_PATH     = "/content/drive/MyDrive/helicopter_data/1class0_train_control.npy"
VAL_DATA_PATH    = "/content/drive/MyDrive/helicopter_data/1class0_test_X.npy"
VAL_CONTROL_PATH = "/content/drive/MyDrive/helicopter_data/1class0_test_control.npy"
AE_PATH          = "/content/iLED/iled/prototsnetresult3/autoencoder_pretrained.pth"
SCALER_PATH      = "/content/iLED/iled/prototsnetresult3/scalers/scalerDebug.pkl"
SAVE_DIR         = "/content/drive/MyDrive/helicopter_data/koopman_checkpoints"

# ★ ARCHITECTURE
NUM_FEATURES     = 314
SEQ_LEN          = 200
LATENT_DIM       = 8      # cycle-scale (must match saved CNN AE latent dim)
TIME_LATENT_DIM  = 6      # timestep-scale (freely chosen)

# ★ MEMORY (time scale only — cycle scale has no memory)
MEMORY_LEN       = 5      # history window L: uses z_{t-L} ... z_{t-1}
MEMORY_HIDDEN    = 32     # hidden channels inside the conv memory kernel
MEMORY_KERNEL_SZ = 3      # Conv1d kernel size inside memory kernel

# ★ TRAINING
BATCH_SIZE_CYCLE = 32
BATCH_SIZE_TIME  = 16     # full trajectories per batch (each is SEQ_LEN=200 steps)
N_EPOCHS         = 750
LR_K             = 3e-3   # Koopman K and B matrices  (physics)
LR_TIME_AE       = 1e-3   # TimeAutoEncoder + memory kernel
LR_ALPHA         = 1e-2   # learnable memory scale
FREEZE_WINDOW_AE = False   # keep pretrained CNN AE frozen throughout
PRETRAIN_EPOCHS  = 150     # phase 1: train time AE only (reconstruction)
KOOPMAN_EPOCHS   = 150    # phase 2: train dynamics + memory, freeze AE
DECODER_WARMUP   = 50
JOINT_EPOCHS     = 450    # phase 3: train everything jointly  (total = 500)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─────────────────────────────────────────────────────────
# iLED-style convolutional memory kernel
# ─────────────────────────────────────────────────────────
class ConvMemoryKernel(nn.Module):
    """
    Causal 1-D convolutional memory kernel, matching the iLED architecture.

    Operates on z_hist: (B, L, d) — the past L latent states strictly
    before the current prediction step.  Because z_hist is always the
    PAST window z_{t-L:t-1}, there is no future leakage.

    Architecture:
        Conv1d(d, hidden, kernel_size, same padding) → SiLU
        Conv1d(hidden, hidden, kernel_size, same padding) → SiLU
        Conv1d(hidden, d, kernel_size=1)   ← pointwise, zero-initialized
        Global average pool over the L axis  → (B, d)

    The final Conv1d is zero-initialized so the memory correction starts
    at exactly zero (Markovian initialization, as in the iLED paper).
    alpha (a separate learnable scalar) scales the whole term so the
    linear Koopman part dominates early in training.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 32,
                 kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding for stride-1 conv
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size, padding=pad),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=1),   # pointwise final
        )
        # Zero-initialize final layer: memory starts as an exact zero correction
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        """
        z_hist : (B, L, d)
        returns : (B, d)
        """
        x = z_hist.permute(0, 2, 1)   # (B, d, L) for Conv1d
        x = self.net(x)                # (B, d, L)
        return x.mean(dim=-1)          # global average pool → (B, d)

def clip_outliers(data, lower=0.5, upper=99.5):
    # data: (N, C, T)
    data = data.copy()
    
    for c in range(data.shape[1]):
        channel_vals = data[:, c, :].reshape(-1)
        
        lo = np.percentile(channel_vals, lower)
        hi = np.percentile(channel_vals, upper)
        
        data[:, c, :] = np.clip(data[:, c, :], lo, hi)
    
    return data

# ─────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────
def load_npz_or_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[list(arr.keys())[0]]
    return arr

sensor_train = load_npz_or_npy(DATA_PATH).transpose(0, 2, 1)     # (N, 314, 200)
sensor_val   = load_npz_or_npy(VAL_DATA_PATH).transpose(0, 2, 1)

sensor_train = clip_outliers(sensor_train, 0.5, 99.5)
sensor_val   = clip_outliers(sensor_val,   0.5, 99.5)



ctrl_train   = load_npz_or_npy(CONTROL_PATH)     # (N, 200, 8)
ctrl_val     = load_npz_or_npy(VAL_CONTROL_PATH)

CTRL_SCALER_SAVE_PATH = "/content/control_scaler.pkl"
CONTROL_DIM = ctrl_train.shape[-1]   # = 8

if os.path.exists(CTRL_SCALER_SAVE_PATH):
    ctrl_scaler = joblib.load(CTRL_SCALER_SAVE_PATH)
    print("Control scaler loaded ✅")
else:
    print("Fitting control scaler...")
    ctrl_scaler = StandardScaler()
    ctrl_scaler.fit(ctrl_train.reshape(-1, CONTROL_DIM))
    joblib.dump(ctrl_scaler, CTRL_SCALER_SAVE_PATH)
    print("Control scaler fitted & saved ✅")

print(ctrl_train.mean(), ctrl_train.std())
print(f"Train sensor : {sensor_train.shape}")
print(f"Val   sensor : {sensor_val.shape}")
print(f"Train ctrl   : {ctrl_train.shape}")
print(f"Val   ctrl   : {ctrl_val.shape}")

assert sensor_train.shape[1] == NUM_FEATURES
assert sensor_train.shape[2] == SEQ_LEN


# ─────────────────────────────────────────────────────────
# 2. Datasets
#
#  CycleDataset    — pairs adjacent full windows: (X_n, X_{n+1})
#                    controls averaged over 200 timesteps → (N-1, 8)
#                    UNCHANGED from original — cycle scale has no memory.
#
#  SequenceDataset — one item = one full trajectory (T=200 timesteps)
#                    'x_seq': (T, 314),  'u_seq': (T, 8)
#                    Required by forward_time because the memory loop
#                    needs to slide z_{t-L:t-1} without crossing
#                    trajectory boundaries.
# ─────────────────────────────────────────────────────────
cycle_train_ds = CycleDataset(sensor_train, ctrl_train)
cycle_val_ds   = CycleDataset(sensor_val,   ctrl_val)

# SequenceDataset takes (N, C, T) channels-first + (N, T, 8) controls
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

print(f"\nCycle  train: {len(cycle_train_ds):>6,} pairs | val: {len(cycle_val_ds):>6,}")
print(f"Time   train: {len(time_train_ds):>6,} trajectories | val: {len(time_val_ds):>6,}")


# ─────────────────────────────────────────────────────────
# 3. Scalers
#    Cycle forward uses (B, 314, T) → sklearn scaler applied per sample
#    Time  forward uses (B*T, 314)  → vectorized StandardScaler
# ─────────────────────────────────────────────────────────
cycle_sklearn_scaler = joblib.load(SCALER_PATH)

# MIN_STD = 1e-1  # tune this (1e-3 to 1e-1 depending on data)

# scales = cycle_sklearn_scaler.scale_

# # clamp
# clamped_scales = np.maximum(scales, MIN_STD)

# # overwrite
# cycle_sklearn_scaler.scale_ = clamped_scales

TIME_SCALER_SAVE_PATH = "/content/time_scaler.pkl"

if os.path.exists(TIME_SCALER_SAVE_PATH):
    time_scaler = joblib.load(TIME_SCALER_SAVE_PATH)
    print("Time scaler loaded ✅")
else:
    print("Fitting time scaler...")
    data_for_scaler = sensor_train.transpose(0, 2, 1).reshape(-1, NUM_FEATURES)
    time_scaler = StandardScaler()
    time_scaler.fit(data_for_scaler)
    joblib.dump(time_scaler, TIME_SCALER_SAVE_PATH)
    print("Time scaler fitted & saved ✅")

_mean  = torch.tensor(time_scaler.mean_,  dtype=torch.float32).to(device)
_scale = torch.tensor(time_scaler.scale_, dtype=torch.float32).to(device)
mean_ts  = _mean.view(1, -1)
scale_ts = _scale.view(1, -1)

_meanc  = torch.tensor(cycle_sklearn_scaler.mean_,  dtype=torch.float32).to(device)
_scalec = torch.tensor(cycle_sklearn_scaler.scale_, dtype=torch.float32).to(device)
mean_cyc  = _meanc.view(1, -1, 1)   # (1, 314, 1)
scale_cyc = _scalec.view(1, -1, 1)


def normalize_cycle_ae(x):
    x_np = x.detach().cpu().numpy()  # (B, C, T)

    x_scaled = transform_ts_data(
        x_np.copy(),
        cycle_sklearn_scaler,
        scale_separately=True,
        fit=False
    )

    return torch.from_numpy(x_scaled).to(x.device)


def denormalize_cycle_ae(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 314, T)  →  original scale (B, 314, T)."""
    x_np = x.detach().cpu().numpy()
    x_inv = np.empty_like(x_np)
    for i in range(x_np.shape[0]):
        x_inv[i] = cycle_sklearn_scaler.inverse_transform(x_np[i].T).T
    return torch.tensor(x_inv, dtype=torch.float32).to(device)


def normalize_time(x: torch.Tensor) -> torch.Tensor:
    """x: (..., 314)  →  normalized."""
    return (x - mean_ts) / (scale_ts + 1e-8)


def denormalize_time(x: torch.Tensor) -> torch.Tensor:
    """x: (..., 314) normalized  →  original scale."""
    return x * scale_ts + mean_ts


def normalize_control(u: torch.Tensor) -> torch.Tensor:
    """
    u: (N, 8) or (B*T, 8) on any device  →  normalized on device.
    Batched: call once with all timesteps concatenated to avoid
    repeated CPU-GPU transfers inside the memory loop.
    """
    u_np = u.detach().cpu().numpy()
    u_scaled = ctrl_scaler.transform(u_np).astype(np.float32)
    return torch.tensor(u_scaled).to(device)


print("Scalers ready ✅")


# ─────────────────────────────────────────────────────────
# 4. Models
# ─────────────────────────────────────────────────────────

# 4a. Cycle AE — pretrained CNN, kept frozen
cycle_ae = MyAutoEncoder(
    num_features=NUM_FEATURES,
    latent_features=LATENT_DIM,
    seq_len=SEQ_LEN,
).to(device)

state_dict = torch.load(AE_PATH, map_location=device)
print("Raw checkpoint keys (first 10):", list(state_dict.keys())[:10])
# Saved from bare RegularConvAutoencoder → add "model." prefix
new_state_dict = {"model." + k: v for k, v in state_dict.items()}
cycle_ae.load_state_dict(new_state_dict)
print("Cycle AE weights loaded ✅")

if FREEZE_WINDOW_AE:
    for p in cycle_ae.parameters():
        p.requires_grad = False
    cycle_ae.eval()
    print("Cycle AE frozen ✅")

# 4b. Time AE — MLP, trained from scratch
time_ae = TimeAutoEncoder(input_dim=NUM_FEATURES, latent_dim=TIME_LATENT_DIM).to(device)

# 4c. Koopman: cycle scale  K(8×8)  B(8×8)
#     Pure linear — no memory.  Cycle scale is UNCHANGED.
cycle_dynamics = KoopmanDynamics(latent_dim=LATENT_DIM, control_dim=CONTROL_DIM).to(device)

# 4d. Koopman: time scale   K(6×6)  B(6×8)
#     Augmented with iLED convolutional memory kernel.
time_dynamics = KoopmanDynamics(latent_dim=TIME_LATENT_DIM, control_dim=CONTROL_DIM).to(device)

# 4e. iLED memory kernel — 1-D CNN over past L latent states (time scale only)
memory_kernel = ConvMemoryKernel(
    latent_dim  = TIME_LATENT_DIM,
    hidden_dim  = MEMORY_HIDDEN,
    kernel_size = MEMORY_KERNEL_SZ,
).to(device)

# 4f. Learnable memory scale alpha — starts at 0.0 (Markovian init)
#     Positive-constrained via softplus in the forward pass so the
#     memory term is always additive and never subtractive.
alpha = nn.Parameter(torch.tensor(0.0, device=device))

memory_len = MEMORY_LEN


for name, module in cycle_ae.named_modules():
    if isinstance(module, torch.nn.BatchNorm1d) or \
       isinstance(module, torch.nn.BatchNorm2d):
        print("BatchNorm found:", name)

# ─────────────────────────────────────────────────────────
# 5. Optimizer
#
#    FIX: cycle_dynamics was missing from the original optimizer,
#    so K_cycle and B_cycle were silently never updated.
#
#    Parameter groups:
#      cycle_dynamics : LR_K     (K_cycle, B_cycle — physics)
#      time_dynamics  : LR_K     (K_time,  B_time  — physics)
#      time_ae        : LR_TIME_AE
#      memory_kernel  : LR_TIME_AE
#      alpha          : LR_ALPHA
#
#    Phase-gating is done via requires_grad flags inside the training
#    loop, not by adding/removing param groups.
# ─────────────────────────────────────────────────────────
optimizer = torch.optim.Adam([
    {'params': cycle_dynamics.parameters(), 'lr': LR_K},
    {'params': time_dynamics.parameters(),  'lr': LR_K},
    {'params': time_ae.parameters(),        'lr': LR_TIME_AE},
    {'params': memory_kernel.parameters(),  'lr': LR_TIME_AE},
    {'params': cycle_ae.parameters(),       'lr': 1e-4},   # ← NEW (smaller LR!)
    {'params': [alpha],                     'lr': LR_ALPHA},
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20
)

# Separate lists for per-group gradient clipping.
# FIX: use lists (not generator expressions) so clip_grad_norm_ can
#      iterate them more than once and cycle_dynamics is included.
k_params   = [cycle_dynamics.K, time_dynamics.K]
b_params   = [p for p in [cycle_dynamics.B, time_dynamics.B] if p is not None]
ae_params  = list(time_ae.parameters()) + list(memory_kernel.parameters()) + [alpha]


# ─────────────────────────────────────────────────────────
# 6. Forward pass functions
# ─────────────────────────────────────────────────────────

def forward_cycle(batch: dict) -> dict:
    """
    Window-scale forward.  UNCHANGED — pure linear Koopman, no memory.

    x_t, x_next : (B, 314, 200)
    u_t          : (B, 8) — window-averaged control
    """
    x_t    = batch['x_t'].to(device)
    x_next = batch['x_next'].to(device)
    u_t    = batch['u_t'].to(device) if 'u_t' in batch else None
    if u_t is not None:
        u_t = normalize_control(u_t)

    #print("RAW x_t stats:", x_t.mean().item(), x_t.std().item())

    xs  = normalize_cycle_ae(x_t)
    xns = normalize_cycle_ae(x_next)

    #print("AFTER scaling:", xs.mean().item(), xs.std().item())

    # x1 = x_t[:1]

    # xs1 = normalize_cycle_ae(x1)
    # x_back = denormalize_cycle_ae(xs1)

    #print("reconstruction error:", (x1 - x_back).abs().mean())

    xs_np = xs.detach().cpu().numpy()

    # means = xs_np.mean(axis=(0,2))
    # stds  = xs_np.std(axis=(0,2))

    # print("mean (first 5):", means[:5])
    # print("std  (first 5):", stds[:5])
    # print("std range:", stds.min(), stds.max())

    z           = cycle_ae.encode(xs)            # (B, 8)
    z_next      = cycle_ae.encode(xns)           # (B, 8)
    z_next_pred = cycle_dynamics(z, u_t)         # (B, 8)  K@z + B@u

    recon_norm      = cycle_ae.decode(z)                # normalized
    recon_pred_norm = cycle_ae.decode(z_next_pred)

    recon      = denormalize_cycle_ae(recon_norm)       # for visualization
    recon_pred = denormalize_cycle_ae(recon_pred_norm)

    return {
        'z': z, 'z_next': z_next, 'z_next_pred': z_next_pred,
        'recon': recon, 'recon_pred': recon_pred,
        'recon_norm': recon_norm,   # ← ADD
        'xs': xs,                   # ← ADD
        'x_t': x_t, 'x_next': x_next,
    }


def forward_time(batch: dict) -> dict:
    """
    Time-scale forward with iLED-style convolutional Mori-Zwanzig memory.

    batch['x_seq'] : (B, T, 314)  — full trajectory, unnormalized
    batch['u_seq'] : (B, T,   8)  — control inputs, unnormalized

    Prediction equation (per step t >= memory_len):
        z_{t+1} = K z_t + B u_t
                + softplus(alpha) * memory_kernel(z_{t-L:t-1})

    Returns:
        preds     : (B, S, d)   predicted latents   S = T - memory_len
        targets   : (B, S, d)   true latents
        xs        : (B, T, 314) normalized sensor input (for recon loss)
        xs_recon  : (B, T, 314) normalized AE reconstruction
        x_t       : (B, T, 314) original unnormalized sensor data
        z_seq     : (B, T, d)   full encoded trajectory
        z_lin_last: (B, d)      linear part at the final prediction step
        z_mem_last: (B, d)      raw memory output at the final step
    """
    x_seq = batch['x_seq'].to(device)   # (B, T, 314)
    u_seq = batch['u_seq'].to(device)   # (B, T, 8)
    B, T, D = x_seq.shape

    # ── Normalize & encode entire trajectory at once ──────────────
    xs = normalize_time(x_seq.view(B * T, D)).view(B, T, D)         # (B, T, D)
    z_seq = time_ae.encode(xs.view(B * T, D)).view(B, T, -1)        # (B, T, d)

    # AE reconstruction in normalized space (used for reconstruction loss)
    xs_recon = time_ae.decode(z_seq.view(B * T, -1)).view(B, T, D)  # (B, T, D)

    # ── Normalize all controls in one batched call ────────────────
    # FIX: original code called normalize_control inside the loop →
    #      T CPU-GPU round-trips per batch.  Doing it once here is
    #      dramatically faster.
    u_norm = normalize_control(u_seq.view(B * T, CONTROL_DIM)).view(B, T, CONTROL_DIM)

    # ── Positive-constrained alpha via softplus ───────────────────
    alpha_pos = torch.nn.functional.softplus(alpha)

    # ── Memory loop ───────────────────────────────────────────────
    preds, targets = [], []
    z_lin_last = z_mem_last = None

    for t in range(memory_len, T):
        z_t  = z_seq[:, t - 1]                       # (B, d)
        u_t  = u_norm[:, t - 1]                      # (B, 8)

        # Linear Koopman term
        z_lin = time_dynamics(z_t, u_t)               # (B, d)

        # iLED memory term: 1-D CNN over the past L latent states
        z_hist = z_seq[:, t - memory_len : t]         # (B, L, d)
        z_mem  = memory_kernel(z_hist)                # (B, d)

        # Full prediction: linear + scaled memory correction
        z_next_pred = z_lin + alpha_pos * z_mem       # (B, d)

        preds.append(z_next_pred)
        targets.append(z_seq[:, t])
        z_lin_last, z_mem_last = z_lin, z_mem

    # AFTER
    preds   = torch.stack(preds,   dim=1)   # (B, S, d)
    targets = torch.stack(targets, dim=1)   # (B, S, d)

    # Decode the PREDICTED latents → sensor space (normalized).
    # This is what the cycle must get right — not just AE round-trip quality.
    B_sz, S, _ = preds.shape
    xs_pred_recon = time_ae.decode(preds.view(B_sz * S, -1)).view(B_sz, S, D)

    return {
        'preds':         preds,
        'targets':       targets,
        'xs':            xs,            # (B, T, D) normalized
        'xs_recon':      xs_recon,      # (B, T, D) AE round-trip quality (GT z → decode)
        'xs_pred_recon': xs_pred_recon, # (B, S, D) dynamics prediction quality (pred z → decode)
        'x_t':           x_seq,         # (B, T, D) original unnormalized
        'z_seq':         z_seq,
        'z_lin_last':    z_lin_last,
        'z_mem_last':    z_mem_last,
    }


# ─────────────────────────────────────────────────────────
# 7. Loss helpers
# ─────────────────────────────────────────────────────────

def koopman_loss_cycle(out: dict, w_latent: float = 1.0,
                       w_recon: float = 1.0) -> torch.Tensor:
    """Latent prediction MSE + (small) reconstruction loss for cycle scale."""
    loss_latent = ((out['z_next_pred'] - out['z_next']) ** 2).mean()
    weights = torch.ones(NUM_FEATURES, device=device)

    # Example: boost important channels
    important_channels = [5,6,7,8,18,19,23,24,25,26,27,28,29,30,31,32,33,34,35,292,293,294,295,296]
    weights[important_channels] = 50.0   # or 50

    # reshape for broadcasting (B, C, T)
    weights = weights.view(1, -1, 1)

    loss_recon = ((out['recon_norm'] - out['xs'])**2 * weights).mean()
    return w_latent * loss_latent + w_recon * loss_recon


def stability_penalty(K: torch.Tensor) -> torch.Tensor:
    """Penalise eigenvalues of K with |λ| > 1 (diverging modes)."""
    return torch.relu(torch.linalg.eigvals(K).abs() - 1.0).mean()


# ─────────────────────────────────────────────────────────
# 8. Training loop
# ─────────────────────────────────────────────────────────
best_val_loss = float('inf')
print(f"\nStarting training — {N_EPOCHS} epochs")
print(f"  Cycle  K({LATENT_DIM}×{LATENT_DIM})  B({LATENT_DIM}×{CONTROL_DIM})"
      f"  LR_K={LR_K}  (no memory)")
print(f"  Time   K({TIME_LATENT_DIM}×{TIME_LATENT_DIM})  B({TIME_LATENT_DIM}×{CONTROL_DIM})"
      f"  LR_K={LR_K}  memory_len={memory_len}")
print(f"  TimeAE LR={LR_TIME_AE}  alpha (learnable) init=0.0\n")

for epoch in range(1, N_EPOCHS + 1):

    # ── Phase scheduling ──────────────────────────────────
    if epoch <= PRETRAIN_EPOCHS:
        phase = "pretrain"
    elif epoch <= PRETRAIN_EPOCHS + KOOPMAN_EPOCHS:
        phase = "koopman"
    elif epoch <= PRETRAIN_EPOCHS + KOOPMAN_EPOCHS + DECODER_WARMUP:
        phase = "decoder_warmup"
    else:
        phase = "joint"

    # ── requires_grad gating ──────────────────────────────
    # pretrain: only time_ae trains (learn a good latent space first)
    # koopman : dynamics + memory train, AE frozen
    # joint   : everything except cycle_ae trains
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
        time_ae.eval()
        cycle_ae.eval()

        for p in time_ae.parameters():        p.requires_grad = False
        for p in cycle_ae.parameters():       p.requires_grad = False
        for p in cycle_dynamics.parameters(): p.requires_grad = True
        for p in time_dynamics.parameters():  p.requires_grad = True
        for p in memory_kernel.parameters():  p.requires_grad = True
        alpha.requires_grad = True

    elif phase == "decoder_warmup":
        # Re-train the AE decoder with dynamics frozen.
        # This forces the decoder to learn to reconstruct from the latents
        # that the Koopman dynamics has already learned to navigate.
        time_ae.train()
        cycle_ae.eval()

        for p in time_ae.parameters():        p.requires_grad = True
        for p in cycle_ae.parameters():       p.requires_grad = False
        for p in cycle_dynamics.parameters(): p.requires_grad = False
        for p in time_dynamics.parameters():  p.requires_grad = False
        for p in memory_kernel.parameters():  p.requires_grad = False
        alpha.requires_grad = False

    elif phase == "joint":
        time_ae.train()
        cycle_ae.train()   # ← IMPORTANT

        for p in time_ae.parameters():        p.requires_grad = True
        for p in cycle_ae.parameters():       p.requires_grad = True
        for p in cycle_dynamics.parameters(): p.requires_grad = True
        for p in time_dynamics.parameters():  p.requires_grad = True
        for p in memory_kernel.parameters():  p.requires_grad = True
        alpha.requires_grad = True

    # ── train mode ────────────────────────────────────────
    cycle_dynamics.train()
    time_dynamics.train()
    memory_kernel.train()
    # if FREEZE_WINDOW_AE:
    cycle_ae.eval()   # keep BN running stats fixed

    tr_cyc, tr_ts = [], []

    # Cycle loader (N-1 pairs) is shorter than time loader (N trajectories)
    # Resample the time iterator to stay in lock-step with cycle batches.
    time_iter = iter(time_train_loader)

    for cyc_batch in cycle_train_loader:

        try:
            ts_batch = next(time_iter)
        except StopIteration:
            time_iter = iter(time_train_loader)
            ts_batch  = next(time_iter)

        optimizer.zero_grad()

        # ── Cycle scale (linear Koopman, no memory) ──────
        out_cyc  = forward_cycle(cyc_batch)
        loss_cyc = koopman_loss_cycle(out_cyc, w_latent=1.5, w_recon=1.0)

        # ── Time scale (linear Koopman + iLED memory) ────
        # AFTER
        out_ts        = forward_time(ts_batch)
        loss_latent   = ((out_ts['preds'] - out_ts['targets']) ** 2).mean()

        # AE round-trip quality: encode GT → decode GT (keeps latent space meaningful)
        loss_recon_ae = ((out_ts['xs_recon'] - out_ts['xs']) ** 2).mean()

        # Dynamics prediction quality: decode PREDICTED z vs GT sensor.
        # This is the critical loss that prevents latent collapse — without it
        # the Koopman operator can trivially satisfy loss_latent by making all
        # latents near-constant while the decoder never has to predict anything.
        loss_pred_sensor = ((out_ts['xs_pred_recon']
                             - out_ts['xs'][:, memory_len:, :]) ** 2).mean()

        # Phase-specific total loss
        if phase == "pretrain":
            # Train time AE via pure reconstruction; Koopman terms frozen
            loss = loss_recon_ae

        elif phase == "koopman":
            # Train dynamics + memory; keep AE quality via round-trip term
            loss = loss_cyc + loss_latent + 0.1 * loss_recon_ae
        

        elif phase == "decoder_warmup":
            # Pure reconstruction from predicted latents — forces decoder alignment
            loss = loss_recon_ae + loss_pred_sensor

        elif phase == "joint":
            loss = (loss_cyc
                    + loss_latent
                    + 0.5 * loss_pred_sensor
                    + 0.05 * loss_recon_ae)   # keeps AE from drifting

        # Stability penalty on both K matrices (skip during pretrain)
        if phase != "pretrain":
            stab = stability_penalty(cycle_dynamics.K) + stability_penalty(time_dynamics.K)
            loss = loss + 5e-4 * stab

        loss.backward()

        # Per-group gradient clipping
        torch.nn.utils.clip_grad_norm_(k_params,  max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(b_params,  max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(ae_params, max_norm=1.0)

        optimizer.step()

        if epoch % 20 == 0:
            print(f"[DEBUG] ||K_cycle||: {cycle_dynamics.K.norm().item():.4f}  "
                  f"alpha (softplus): {torch.nn.functional.softplus(alpha).item():.4f}")

        # AFTER
        if phase in ("pretrain", "decoder_warmup"):
            tr_cyc.append(0.0)
            tr_ts.append(loss_recon_ae.detach().item())
        else:
            tr_cyc.append(loss_cyc.detach().item())
            tr_ts.append(loss_pred_sensor.detach().item())

    # ── Evaluation ────────────────────────────────────────
    cycle_dynamics.eval()
    time_dynamics.eval()
    time_ae.eval()
    memory_kernel.eval()
    cycle_ae.eval()

    va_cyc, va_ts, va_recon = [], [], []

    with torch.no_grad():

        # ---- Latent distribution debug (every 10 epochs) ----
        if epoch % 10 == 0:
            train_batch = next(iter(cycle_train_loader))
            out_tr = forward_cycle(train_batch)
            val_batch_c = next(iter(cycle_val_loader))
            out_va = forward_cycle(val_batch_c)

            print(f"\n[DEBUG] Cycle TRAIN latent — mean: {out_tr['z'].mean().item():.4f}"
                  f"  std: {out_tr['z'].std().item():.4f}")
            print(f"[DEBUG] Cycle VAL   latent — mean: {out_va['z'].mean().item():.4f}"
                  f"  std: {out_va['z'].std().item():.4f}")
            tr_err = ((out_tr['z_next_pred'] - out_tr['z_next'])**2).mean().item()
            va_err = ((out_va['z_next_pred'] - out_va['z_next'])**2).mean().item()
            print(f"[DEBUG] Cycle latent MSE — train: {tr_err:.4e}  val: {va_err:.4e}")
            print(f"[DEBUG] ||z|| train: {out_tr['z'].norm(dim=1).mean().item():.3f}  "
                  f"||z_pred|| train: {out_tr['z_next_pred'].norm(dim=1).mean().item():.3f}")
            # Latent variance check — std should be >> 0.
            # If std < 0.01 across the trajectory, the latent space has collapsed.
            z_std = out_val_t['z_seq'].std(dim=1).mean().item()
            print(f"[COLLAPSE CHECK] z_seq std (should be > 0.1): {z_std:.4f}  "
                  f"alpha={torch.nn.functional.softplus(alpha).item():.4f}")
            print("=" * 60)

        # ---- Cycle val loss ----
        for i, batch in enumerate(cycle_val_loader):
            loss_val = koopman_loss_cycle(forward_cycle(batch)).item()
            va_cyc.append(loss_val)
            if epoch % 20 == 0 and i < 3:
                print(f"[DEBUG] val cyc batch {i}: {loss_val:.4e}")

        # ---- Time val loss ----
        # FIX: was calling koopman_loss_cycle() on time output (wrong keys)
        #      and reading 'recon'/'x_t' which forward_time didn't return.
        # AFTER
        for batch in time_val_loader:
            out_t = forward_time(batch)
            va_ts.append(((out_t['preds'] - out_t['targets'])**2).mean().item())

            # Use the sensor-prediction loss for validation too, so the
            # "best" checkpoint reflects real dynamics quality, not AE quality.
            pred_sensor_loss = ((out_t['xs_pred_recon']
                                 - out_t['xs'][:, memory_len:, :])**2).mean().item()
            va_recon.append(pred_sensor_loss)

        # ---- Linear vs memory norm debug (once per epoch) ----
        # FIX: original block referenced undefined T and 'u_seq' key.
        val_batch_t = next(iter(time_val_loader))
        out_val_t   = forward_time(val_batch_t)
        alpha_pos   = torch.nn.functional.softplus(alpha)
        if epoch % 10 == 0:

            if out_val_t['z_lin_last'] is not None:
                lin_norm = out_val_t['z_lin_last'].norm(dim=-1).mean()
                mem_norm = (alpha_pos * out_val_t['z_mem_last']).norm(dim=-1).mean()
                print(f"[DEBUG] ||z_lin|| ≈ {lin_norm:.3f}  "
                    f"||α·z_mem|| ≈ {mem_norm:.3f}  "
                    f"ratio={mem_norm / (lin_norm + 1e-6):.3f}  "
                    f"α={alpha_pos.item():.4f}")

            # ---- First 3 val batch MSEs ----
            for i, batch in enumerate(time_val_loader):
                if i >= 3:
                    break
                out_t = forward_time(batch)
                loss_i = ((out_t['preds'] - out_t['targets'])**2).mean().item()
                print(f"[VAL DEBUG] Time batch {i} latent MSE = {loss_i:.3e}")

    # ── Epoch summary ─────────────────────────────────────
    tr_c = np.mean(tr_cyc);  tr_t = np.mean(tr_ts)
    va_c = np.mean(va_cyc);  va_t = np.mean(va_ts)
    va_total = va_c + va_t
    val_recon_loss = np.mean(va_recon)

    scheduler.step(va_total)

    improved = va_total < best_val_loss
    print(
        f"Epoch {epoch:4d}/{N_EPOCHS} | phase={phase:8s} | "
        f"cyc  tr {tr_c:.3e} va {va_c:.3e} | "
        f"ts   tr {tr_t:.3e} va {va_t:.3e} | "
        f"recon {val_recon_loss:.3e}"
        + (" ← best" if improved else "")
    )

    if improved:
        best_val_loss = va_total
        torch.save({
            'epoch':          epoch,
            'cycle_dynamics': cycle_dynamics.state_dict(),
            'time_dynamics':  time_dynamics.state_dict(),
            'time_ae':        time_ae.state_dict(),
            'cycle_ae':       cycle_ae.state_dict(),   # ← ADD THIS
            'memory_kernel':  memory_kernel.state_dict(),
            'alpha':          alpha.data,
            'optimizer':      optimizer.state_dict(),
            'val_loss':       best_val_loss,
            'K_cycle':        cycle_dynamics.K.detach().cpu(),
            'K_time':         time_dynamics.K.detach().cpu(),
        }, os.path.join(SAVE_DIR, "best.pth"))

    if epoch % 50 == 0:
        torch.save({
            'epoch':          epoch,
            'cycle_dynamics': cycle_dynamics.state_dict(),
            'time_dynamics':  time_dynamics.state_dict(),
            'time_ae':        time_ae.state_dict(),
            'memory_kernel':  memory_kernel.state_dict(),
            'alpha':          alpha.data,
        }, os.path.join(SAVE_DIR, f"epoch_{epoch}.pth"))


# ─────────────────────────────────────────────────────────
# 9. Final summary
# ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training complete. Best val loss: {best_val_loss:.4e}")
print(f"Checkpoints saved to: {SAVE_DIR}")
print(f"Learned alpha (memory scale, softplus): "
      f"{torch.nn.functional.softplus(alpha).item():.4f}")

for label, dyn in [
    (f"Cycle  ({LATENT_DIM}×{LATENT_DIM})", cycle_dynamics),
    (f"Time   ({TIME_LATENT_DIM}×{TIME_LATENT_DIM})", time_dynamics),
]:
    K = dyn.K.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(K)
    print(f"\n{label} K:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print(f"  Eigenvalues    : {eigvals}")
    print(f"  Stable (|λ|≤1) : {all(abs(e) <= 1 for e in eigvals)}")
    if dyn.B is not None:
        print(f"  B norm (ctrl)  : {dyn.B.norm().item():.4f}")