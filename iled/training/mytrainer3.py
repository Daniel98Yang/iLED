# mytrainer3.py
#
# Two-scale Koopman training:
#   CYCLE scale  — pretrained CNN AE (frozen), adjacent window pairs,    latent dim = 3
#   TIME  scale  — trainable MLP AE,           adjacent timestep pairs,  latent dim = 6
#
# Control (8 channels) enters only through the B matrix.
# K (physics) and B (control) are in separate optimizer groups at different LRs.
# Memory: DISABLED.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import joblib
import os

from myautoencoder   import MyAutoEncoder
from smallscaleae    import TimeAutoEncoder
from koopmandataset  import KoopmanDataset, CycleDataset
from koopmandynamics import KoopmanDynamics
from sklearn.preprocessing import StandardScaler
from datasets_utils import transform_ts_data

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ PATHS
DATA_PATH        = "/content/drive/MyDrive/helicopter_data/class1_train.npy"
CONTROL_PATH     = "/content/drive/MyDrive/helicopter_data/control_class1_train.npz"
VAL_DATA_PATH    = "/content/drive/MyDrive/helicopter_data/class1_test.npy"
VAL_CONTROL_PATH = "/content/drive/MyDrive/helicopter_data/control_class1_test.npz"
AE_PATH          = "/content/iLED/iled/prototsnetresult/autoencoder_pretrained.pth"
SCALER_PATH      = "/content/iLED/iled/prototsnetresult/scalers/s2_pf3_pc1_pl0.5_cl0.04_sp-0.05_scaler.pkl"
SAVE_DIR         = "/content/drive/MyDrive/helicopter_data/koopman_checkpoints"

# ★ ARCHITECTURE
NUM_FEATURES     = 314
SEQ_LEN          = 200
LATENT_DIM       = 3      # cycle-scale (must match saved CNN AE)
TIME_LATENT_DIM  = 6      # timestep-scale (freely chosen)

# ★ TRAINING
BATCH_SIZE_CYCLE = 32
BATCH_SIZE_TIME  = 256    # larger batch is fine — many more time-pairs
N_EPOCHS         = 500
LR_K             = 1e-3   # Koopman K matrices  (physics)
LR_B             = 1e-4   # Koopman B matrices  (control — more conservative)
LR_TIME_AE       = 1e-3   # TimeAutoEncoder     (trained jointly)
FREEZE_WINDOW_AE = True   # keep pretrained CNN AE frozen
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────
def load_npz_or_npy(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[list(arr.keys())[0]]
    return arr

sensor_train = load_npz_or_npy(DATA_PATH)        # (N, 314, 200)
sensor_val   = load_npz_or_npy(VAL_DATA_PATH)

# Controls shape must be (N, 200, 8) — full 200 timesteps, NOT pre-trimmed
# Trimming (to align pairs) is handled inside each Dataset class
ctrl_train   = load_npz_or_npy(CONTROL_PATH)     # (N, 200, 8)
ctrl_val     = load_npz_or_npy(VAL_CONTROL_PATH)

print(ctrl_train.mean(), ctrl_train.std())

print(f"Train sensor : {sensor_train.shape}")
print(f"Val   sensor : {sensor_val.shape}")
print(f"Train ctrl   : {ctrl_train.shape}")
print(f"Val   ctrl   : {ctrl_val.shape}")

assert sensor_train.shape[1] == NUM_FEATURES
assert sensor_train.shape[2] == SEQ_LEN
CONTROL_DIM = ctrl_train.shape[-1]  # = 8

# ─────────────────────────────────────────────────────────
# 2. Datasets
#
#  CycleDataset  — pairs adjacent full windows: (X_n, X_{n+1})
#                  controls averaged over 200 timesteps → (N-1, 8)
#
#  KoopmanDataset — pairs adjacent timesteps within each window
#                   data passed as (N, 200, 314) i.e. (N, T, features)
#                   controls passed as (N, 200, 8) — Dataset trims to 199 internally
# ─────────────────────────────────────────────────────────
cycle_train_ds = CycleDataset(sensor_train, ctrl_train)
cycle_val_ds   = CycleDataset(sensor_val,   ctrl_val)

# KoopmanDataset expects (N, T, features) — transpose from (N, 314, 200)
time_train_ds  = KoopmanDataset(sensor_train.transpose(0, 2, 1), controls=ctrl_train)
time_val_ds    = KoopmanDataset(sensor_val.transpose(0, 2, 1),   controls=ctrl_val)

cycle_train_loader = DataLoader(cycle_train_ds, batch_size=BATCH_SIZE_CYCLE, shuffle=True,  num_workers=2)
cycle_val_loader   = DataLoader(cycle_val_ds,   batch_size=BATCH_SIZE_CYCLE, shuffle=False, num_workers=2)
time_train_loader  = DataLoader(time_train_ds,  batch_size=BATCH_SIZE_TIME,  shuffle=True,  num_workers=2)
time_val_loader    = DataLoader(time_val_ds,    batch_size=BATCH_SIZE_TIME,  shuffle=False, num_workers=2)

print(f"\nCycle  train: {len(cycle_train_ds):>6,} pairs | val: {len(cycle_val_ds):>6,}")
print(f"Time   train: {len(time_train_ds):>6,} pairs | val: {len(time_val_ds):>6,}")

# ─────────────────────────────────────────────────────────
# 3. Scaler
#    Cycle forward uses (B, 314, T) → broadcast shape (1, 314, 1)
#    Time  forward uses (B, 314)    → broadcast shape (1, 314)
# ─────────────────────────────────────────────────────────
cycle_sklearn_scaler = joblib.load(SCALER_PATH)

TIME_SCALER_SAVE_PATH = "/content/time_scaler.pkl"

if os.path.exists(TIME_SCALER_SAVE_PATH):
    time_scaler = joblib.load(TIME_SCALER_SAVE_PATH)
    print("Time scaler loaded ✅")
else:
    print("Fitting time scaler...")

    # reshape (N, 314, 200) → (N*200, 314)
    data_for_scaler = sensor_train.transpose(0, 2, 1).reshape(-1, NUM_FEATURES)

    time_scaler = StandardScaler()
    time_scaler.fit(data_for_scaler)

    joblib.dump(time_scaler, TIME_SCALER_SAVE_PATH)
    print("Time scaler fitted & saved ✅")

_mean  = torch.tensor(time_scaler.mean_,  dtype=torch.float32).to(device)
_scale = torch.tensor(time_scaler.scale_, dtype=torch.float32).to(device)

mean_ts   = _mean.view(1, -1)
scale_ts  = _scale.view(1, -1)

_meanc  = torch.tensor(cycle_sklearn_scaler.mean_,  dtype=torch.float32).to(device)
_scalec = torch.tensor(cycle_sklearn_scaler.scale_, dtype=torch.float32).to(device)

mean_cyc  = _meanc.view(1, -1, 1)   # (1, 314, 1)
scale_cyc = _scalec.view(1, -1, 1)


def normalize_cycle_ae(x):
    # x: (B, 314, T)
    x_np = x.detach().cpu().numpy()
    
    x_scaled = np.stack([
        transform_ts_data(sample.copy(), cycle_sklearn_scaler, scale_separately=True, fit=False)
        for sample in x_np
    ])
    
    return torch.tensor(x_scaled, dtype=torch.float32).to(device)


def denormalize_cycle_ae(x):
    x_np = x.detach().cpu().numpy()
    
    x_inv = np.stack([
        cycle_sklearn_scaler.inverse_transform(sample.T).T
        for sample in x_np
    ])
    
    return torch.tensor(x_inv, dtype=torch.float32).to(device)

def normalize_time(x):    return (x - mean_ts)   / (scale_ts  + 1e-8)
def denormalize_time(x):  return x * scale_ts    + mean_ts

print("Scaler loaded ✅")

# ─────────────────────────────────────────────────────────
# 4. Models
# ─────────────────────────────────────────────────────────

# 4a. Cycle AE — pretrained CNN, frozen
cycle_ae = MyAutoEncoder(
    num_features=NUM_FEATURES,
    latent_features=LATENT_DIM,
    seq_len=SEQ_LEN,
).to(device)

state_dict = torch.load(AE_PATH, map_location=device)
print(list(state_dict.keys())[:10])
# If saved from bare RegularConvAutoencoder, add "model." prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = "model." + k   # critical fix
    new_state_dict[new_key] = v

cycle_ae.load_state_dict(new_state_dict)
print("Cycle AE weights loaded ✅")

if FREEZE_WINDOW_AE:
    for p in cycle_ae.parameters():
        p.requires_grad = False
    cycle_ae.eval()
    print("Cycle AE frozen")

# 4b. Time AE — MLP, trained from scratch
time_ae = TimeAutoEncoder(input_dim=NUM_FEATURES, latent_dim=TIME_LATENT_DIM).to(device)

# 4c. Koopman: cycle scale  K(3×3)  B(3×8)
cycle_dynamics = KoopmanDynamics(latent_dim=LATENT_DIM,      control_dim=CONTROL_DIM).to(device)

# 4d. Koopman: time scale   K(6×6)  B(6×8)
time_dynamics  = KoopmanDynamics(latent_dim=TIME_LATENT_DIM, control_dim=CONTROL_DIM).to(device)

# ─────────────────────────────────────────────────────────
# 5. Optimizer — separate groups for K, B, and TimeAE
#    K (physics): LR_K       — standard
#    B (control): LR_B       — 10x lower, control signal harder to learn
#    TimeAE:      LR_TIME_AE — same as K
# ─────────────────────────────────────────────────────────
k_params = [cycle_dynamics.K, time_dynamics.K]
b_params = [p for p in [cycle_dynamics.B, time_dynamics.B] if p is not None]

optimizer = torch.optim.Adam([
    {'params': k_params,                          'lr': LR_K,       'name': 'K_physics'},
    {'params': b_params,                          'lr': LR_B,       'name': 'B_control'},
    {'params': list(time_ae.parameters()),        'lr': LR_TIME_AE, 'name': 'time_ae'},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20
)

# ─────────────────────────────────────────────────────────
# 6. Forward pass functions — one per scale
# ─────────────────────────────────────────────────────────

def forward_cycle(batch):
    """
    Window-scale forward.
    x_t, x_next : (B, 314, 200)
    u_t          : (B, 8) — one averaged control vector per window
    """
    x_t    = batch['x_t'].to(device)
    x_next = batch['x_next'].to(device)
    u_t    = batch['u_t'].to(device) if 'u_t' in batch else None

    xs  = normalize_cycle_ae(x_t)
    xns = normalize_cycle_ae(x_next)

    z           = cycle_ae.encode(xs)            # (B, 3)
    z_next      = cycle_ae.encode(xns)           # (B, 3)  target
    z_next_pred = cycle_dynamics(z, u_t)         # (B, 3)  K@z + B@u

    recon       = denormalize_cycle_ae(cycle_ae.decode(z))
    recon_pred  = denormalize_cycle_ae(cycle_ae.decode(z_next_pred))

    print("x_t mean/std:", x_t.mean().item(), x_t.std().item())
    print("recon mean/std:", recon.mean().item(), recon.std().item())
    print("z mean:", z.abs().mean().item())

    return {'z': z, 'z_next': z_next, 'z_next_pred': z_next_pred,
            'recon': recon, 'recon_pred': recon_pred,
            'x_t': x_t, 'x_next': x_next}


def forward_time(batch):
    """
    Timestep-scale forward.
    x_t, x_next : (B, 314)   single sensor snapshots
    u_t          : (B, 8)    control at that exact timestep
    """
    x_t    = batch['x_t'].to(device)
    x_next = batch['x_next'].to(device)
    u_t    = batch['u_t'].to(device) if 'u_t' in batch else None

    xs  = normalize_time(x_t)
    xns = normalize_time(x_next)

    z           = time_ae.encode(xs)             # (B, 6)
    z_next      = time_ae.encode(xns)            # (B, 6)  target
    z_next_pred = time_dynamics(z, u_t)          # (B, 6)  K@z + B@u

    recon       = denormalize_time(time_ae.decode(z))
    recon_pred  = denormalize_time(time_ae.decode(z_next_pred))

    return {'z': z, 'z_next': z_next, 'z_next_pred': z_next_pred,
            'recon': recon, 'recon_pred': recon_pred,
            'x_t': x_t, 'x_next': x_next}

# ─────────────────────────────────────────────────────────
# 7. Loss helpers
# ─────────────────────────────────────────────────────────

def koopman_loss(out, w_latent=1.0, w_recon=0.5):
    """Latent prediction loss + reconstruction loss."""
    loss_latent = ((out['z_next_pred'] - out['z_next']) ** 2).mean()
    loss_recon  = ((out['recon'] - out['x_t']) ** 2).mean()*0.5 #HYPERPARAMETER - WEIGHTING OF THISf
    return w_latent * loss_latent + w_recon * loss_recon


def stability_penalty(K):
    """Penalise eigenvalues of K with |λ| > 1 (diverging dynamics)."""
    return torch.relu(torch.linalg.eigvals(K).abs() - 1.0).mean()

# ─────────────────────────────────────────────────────────
# 8. Training loop
# ─────────────────────────────────────────────────────────
best_val_loss = float('inf')
print(f"\nStarting training — {N_EPOCHS} epochs")
print(f"  Cycle  K({LATENT_DIM}×{LATENT_DIM})      B({LATENT_DIM}×{CONTROL_DIM})      LR_K={LR_K}  LR_B={LR_B}")
print(f"  Time   K({TIME_LATENT_DIM}×{TIME_LATENT_DIM})      B({TIME_LATENT_DIM}×{CONTROL_DIM})      LR_K={LR_K}  LR_B={LR_B}")
print(f"  TimeAE LR={LR_TIME_AE}\n")

for epoch in range(1, N_EPOCHS + 1):

    # ── train mode ──────────────────────────────────────
    cycle_dynamics.train()
    time_dynamics.train()
    time_ae.train()
    if FREEZE_WINDOW_AE:
        cycle_ae.eval()   # keep frozen AE in eval (BN running stats fixed)

    tr_cyc, tr_ts = [], []

    # Cycle loader is shorter (N-1 pairs vs N*199 pairs)
    # Resample time loader to match cycle steps per epoch
    time_iter = iter(time_train_loader)

    for cyc_batch in cycle_train_loader:

        # Refill time iterator if exhausted
        try:
            ts_batch = next(time_iter)
        except StopIteration:
            time_iter = iter(time_train_loader)
            ts_batch  = next(time_iter)

        optimizer.zero_grad()

        out_cyc = forward_cycle(cyc_batch)
        out_ts  = forward_time(ts_batch)

        loss_cyc  = koopman_loss(out_cyc, w_latent=1.0, w_recon=0.5)
        loss_ts   = koopman_loss(out_ts,  w_latent=0.5, w_recon=0.2)

        stab = (stability_penalty(cycle_dynamics.K) +
                stability_penalty(time_dynamics.K))

        loss = loss_cyc + loss_ts + 1e-3 * stab
        loss.backward()

        # K and B clipped separately — tighter clip for B
        torch.nn.utils.clip_grad_norm_(k_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(b_params, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(time_ae.parameters(), max_norm=1.0)

        optimizer.step()
        tr_cyc.append(loss_cyc.item())
        tr_ts.append(loss_ts.item())

    # ── eval mode ───────────────────────────────────────
    cycle_dynamics.eval()
    time_dynamics.eval()
    time_ae.eval()
    cycle_ae.eval()

    va_cyc, va_ts = [], []

    with torch.no_grad():
        for batch in cycle_val_loader:
            va_cyc.append(koopman_loss(forward_cycle(batch)).item())
        for batch in time_val_loader:
            va_ts.append(koopman_loss(forward_time(batch), w_latent=0.5, w_recon=0.2).item())

    tr_c = np.mean(tr_cyc);  tr_t = np.mean(tr_ts)
    va_c = np.mean(va_cyc);  va_t = np.mean(va_ts)
    va_total = va_c + va_t

    scheduler.step(va_total)

    improved = va_total < best_val_loss
    print(
        f"Epoch {epoch:4d}/{N_EPOCHS} | "
        f"cyc  tr {tr_c:.3e} va {va_c:.3e} | "
        f"ts   tr {tr_t:.3e} va {va_t:.3e}"
        + (" ← best" if improved else "")
    )

    if improved:
        best_val_loss = va_total
        torch.save({
            'epoch':          epoch,
            'cycle_dynamics': cycle_dynamics.state_dict(),
            'time_dynamics':  time_dynamics.state_dict(),
            'time_ae':        time_ae.state_dict(),
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
        }, os.path.join(SAVE_DIR, f"epoch_{epoch}.pth"))

# ─────────────────────────────────────────────────────────
# 9. Final summary
# ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training complete. Best val loss: {best_val_loss:.4e}")
print(f"Checkpoints: {SAVE_DIR}")

for label, dyn in [("Cycle  (3×3)", cycle_dynamics), ("Time   (6×6)", time_dynamics)]:
    K = dyn.K.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(K)
    print(f"\n{label} K:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print(f"  Eigenvalues      : {eigvals}")
    print(f"  Stable (|λ|≤1)   : {all(abs(e) <= 1 for e in eigvals)}")
    if dyn.B is not None:
        print(f"  B norm (control) : {dyn.B.norm().item():.4f}")