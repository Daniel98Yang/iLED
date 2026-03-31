# mytrainer3.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import joblib
import os

from myautoencoder   import MyAutoEncoder
from smallscaleae    import TimeAutoEncoder
from koopmandataset  import KoopmanDataset, CycleDataset
from koopmandynamics import KoopmanDynamics
from losslib         import KoopmanLoss
from timeAttention   import TemporalAttention

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ PATHS
DATA_PATH   = "/content/drive/MyDrive/helicopter_data/class1_train.npy"
CONTROL_PATH = "/content/drive/MyDrive/helicopter_data/control_class1_train.npz"
AE_PATH     = "/content/iLED/iled/prototsnetresult/autoencoder_pretrained.pth"
SCALER_PATH = "/content/iLED/iled/prototsnetresult/scalers/s2_pf3_pc1_pl0.5_cl0.04_sp-0.05_scaler.pkl"
SAVE_DIR    = "/content/drive/MyDrive/helicopter_data/koopman_checkpoints"

# ★ ARCHITECTURE
NUM_FEATURES = 314
LATENT_DIM   = 3
SEQ_LEN      = 200


# ★ TRAINING
BATCH_SIZE = 32
LR         = 1e-3
N_EPOCHS   = 500
VAL_SPLIT  = 0.1
FREEZE_AE  = True   # True = only train K matrix, AE weights are frozen

# Pair adjacent timesteps within each trajectory (recommended for 257 independent flights)
SEQUENTIAL_TRAJECTORIES = True
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load data ──────────────────────────────────────────
data = np.load(DATA_PATH)           # (257, 314, 200)

if CONTROL_PATH:
    controls = np.load(CONTROL_PATH)
    if isinstance(controls, np.lib.npyio.NpzFile):
        controls = controls[list(controls.keys())[0]]

    # ensure shape (N, T-1, control_dim)
    controls = controls[:, :-1, :]
else:
    controls = None

CONTROL_DIM  = controls.shape[-1]

print(f"Data shape: {data.shape}")
print(f"Controls shape: {controls.shape}")
assert data.shape[1] == NUM_FEATURES, f"Expected {NUM_FEATURES} channels, got {data.shape[1]}"
assert data.shape[2] == SEQ_LEN,      f"Expected {SEQ_LEN} timesteps, got {data.shape[2]}"

# ── 2. Dataset and loaders ────────────────────────────────
# ── 2. Dataset and loaders ────────────────────────────────

# ===== TRAIN DATA =====
cycle_dataset = CycleDataset(data, controls)
time_dataset  = KoopmanDataset(data.transpose(0,2,1), controls=controls)

cycle_loader = DataLoader(cycle_dataset, batch_size=BATCH_SIZE, shuffle=True)
time_loader  = DataLoader(time_dataset,  batch_size=BATCH_SIZE, shuffle=True)

# ===== VALIDATION / TEST DATA =====
VAL_DATA_PATH    = "/content/drive/MyDrive/helicopter_data/class1_test.npy"
VAL_CONTROL_PATH = "/content/drive/MyDrive/helicopter_data/control_class1_test.npz"

val_data = np.load(VAL_DATA_PATH)

val_controls = np.load(VAL_CONTROL_PATH)
if isinstance(val_controls, np.lib.npyio.NpzFile):
    val_controls = val_controls[list(val_controls.keys())[0]]

val_controls = val_controls[:, :-1, :]

# build datasets
val_cycle_dataset = CycleDataset(val_data, val_controls)
val_time_dataset  = KoopmanDataset(val_data.transpose(0,2,1), controls=val_controls)

# loaders
val_cycle_loader = DataLoader(val_cycle_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_time_loader  = DataLoader(val_time_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train cycle: {len(cycle_dataset):,} | Train time: {len(time_dataset):,}")
print(f"Val cycle:   {len(val_cycle_dataset):,} | Val time:   {len(val_time_dataset):,}")


# ── 3. Scaler (sklearn → torch, lives on device) ─────────
sklearn_scaler = joblib.load(SCALER_PATH)
scaler_mean  = torch.tensor(sklearn_scaler.mean_,  dtype=torch.float32).view(1, -1, 1).to(device)
scaler_scale = torch.tensor(sklearn_scaler.scale_, dtype=torch.float32).view(1, -1, 1).to(device)

def scale(x):
    """Normalize: (B, 314, T) → (B, 314, T)"""
    return (x - scaler_mean) / (scaler_scale + 1e-8)

def unscale(x):
    """Denormalize: (B, 314, T) → (B, 314, T)"""
    return x * scaler_scale + scaler_mean

print("Scaler loaded ✅")

def build_memory(z_seq, k=3):
    """
    z_seq: (B, latent_dim)
    returns memory vector (B, k*latent_dim)
    """
    B, D = z_seq.shape
    mem = z_seq.repeat(1, k)   # simple placeholder
    return mem

# ── 4. Autoencoder ────────────────────────────────────────
ae = MyAutoEncoder(
    num_features=NUM_FEATURES,
    latent_features=LATENT_DIM,
    seq_len=SEQ_LEN,
).to(device)

state = torch.load(AE_PATH, map_location=device, weights_only=False)
ae.load_state_dict(state)
print("AE weights loaded ✅")

# ── 4b. Time-scale autoencoder (NEW) ─────────────────────
TIME_LATENT_DIM = 6

time_ae = TimeAutoEncoder(
    input_dim=NUM_FEATURES,
    latent_dim=TIME_LATENT_DIM
).to(device)



if FREEZE_AE:
    for p in ae.parameters():
        p.requires_grad = False
    # for o in time_ae.parameters():
    #     o.requires_grad = False
    ae.eval()
    # time_ae.eval()
    print("AE frozen — training Koopman K matrix only")

# ── 5. Koopman dynamics ───────────────────────────────────
dynamics = KoopmanDynamics(
    latent_dim=LATENT_DIM,
    control_dim=CONTROL_DIM,
    use_memory=False   # ← this is your iLED closure
).to(device)
print(f"KoopmanDynamics: K is ({LATENT_DIM}×{LATENT_DIM})")

time_dynamics = KoopmanDynamics(
    latent_dim=TIME_LATENT_DIM,
    control_dim=CONTROL_DIM,
    use_memory=False
).to(device)

attn = TemporalAttention(TIME_LATENT_DIM).to(device)
proj = nn.Linear(TIME_LATENT_DIM, LATENT_DIM).to(device)

# ── 6. Optimizer and loss ─────────────────────────────────
# Only train Koopman parameters (AE frozen)
optimizer = torch.optim.Adam(
    list(dynamics.parameters()) +
    list(time_dynamics.parameters()) +
    list(time_ae.parameters()) +
    list(attn.parameters()) +
    list(proj.parameters()),
    lr=LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, verbose=True
)
loss_fn = KoopmanLoss(recon_scale=1.0, latent_scale=1.0, forecast_scale=0.5)

# ── 7. Forward pass function ──────────────────────────────
def run_forward(batch):
    x_t    = batch['x_t'].to(device)      # (B, 314, T)
    x_next = batch['x_next'].to(device)   # (B, 314, T)

    u_t = batch['u_t'].to(device)

    # Scale in sensor space
    x_t_sc    = scale(x_t)
    x_next_sc = scale(x_next)

    # ===== Cycle-scale (existing) =====
    z      = ae.encode(x_t_sc)       # (B, 3)
    z_next = ae.encode(x_next_sc)

    z_next_pred = dynamics(z, u_t)

    # ===== Time-scale (NEW) =====
    # reshape: (B, 314, T) → (B*T, 314)
    B, C, T = x_t_sc.shape

    x_t_steps    = x_t_sc.permute(0, 2, 1).reshape(-1, C)
    x_next_steps = x_next_sc.permute(0, 2, 1).reshape(-1, C)

    # encode per timestep
    z_t      = time_ae.encode(x_t_steps)        # (B*T, d)
    z_next_t = time_ae.encode(x_next_steps)

    # reshape back
    z_t      = z_t.view(B, T, -1)
    z_next_t = z_next_t.view(B, T, -1)

    # Koopman on time-scale
    u_t_expanded = u_t.unsqueeze(1).repeat(1, T, 1).reshape(-1, u_t.shape[-1])

    z_t_flat = z_t.reshape(-1, z_t.shape[-1])
    z_t_pred = time_dynamics(z_t_flat, u_t_expanded)
    z_t_pred = z_t_pred.view(B, T, -1)

    # ===== Coupling =====
    # ===== Attention aggregation =====
    z_t_attn = attn(z_t)   # (B, TIME_LATENT_DIM)

    # ===== Projection to cycle latent =====
    z_t_proj = proj(z_t_attn)   # (B, LATENT_DIM)

    # Decode for reconstruction losses (back to sensor space)
    recon          = unscale(ae.decode(z))           # (B, 314, T)
    recon_forecast = unscale(ae.decode(z_next_pred)) # (B, 314, T)

    linear_part, nl_part  = dynamics.evaluate_dynamics_parts(z, u_t)

    return {
        "z":                      z,
        "z_next":                 z_next,
        "z_next_pred":            z_next_pred,
        "z_t": z_t,
        "z_t_next": z_next_t,
        "z_t_pred": z_t_pred,
        "z_t_attn": z_t_attn,
        "z_t_proj": z_t_proj,
        "reconstruction":         recon,
        "reconstructed_forecast": recon_forecast,
        "dynamics_parts":         (linear_part, None),
        # losslib needs x_t in batch — we add it back below
    }, x_t, x_next, u_t

# ── 8. Training loop ──────────────────────────────────────
best_val_loss = float('inf')
print(f"\nStarting training for {N_EPOCHS} epochs...\n")

for epoch in range(1, N_EPOCHS + 1):

    # --- Train ---
    dynamics.train()
    train_losses = []

    for (batch_time, batch_cycle) in zip(time_loader, cycle_loader):

        optimizer.zero_grad()

        # ===== TIME SCALE =====
        x_t_time    = batch_time['x_t'].to(device)
        x_next_time = batch_time['x_next'].to(device)
        u_time      = batch_time['u_t'].to(device)

        z_t      = time_ae.encode(x_t_time)
        z_next_t = time_ae.encode(x_next_time)
        z_pred_t = time_dynamics(z_t, u_time)

        # ===== CYCLE SCALE =====
        x_t_cycle    = batch_cycle['x_t'].to(device)
        x_next_cycle = batch_cycle['x_next'].to(device)
        u_cycle      = batch_cycle['u_t'].to(device)

        x_t_sc    = scale(x_t_cycle)
        x_next_sc = scale(x_next_cycle)

        z_c      = ae.encode(x_t_sc)
        z_next_c = ae.encode(x_next_sc)
        z_pred_c = dynamics(z_c, u_cycle)

        # ===== COUPLING =====
        B = x_t_cycle.shape[0]
        T = x_t_cycle.shape[2]

        # reshape timestep latent into sequence
        z_t_seq = z_t.view(B, -1, z_t.shape[-1])

        z_t_attn = attn(z_t_seq)
        z_t_proj = proj(z_t_attn)

        # ===== LOSSES =====
        loss_time   = torch.mean((z_pred_t - z_next_t)**2)
        loss_cycle  = torch.mean((z_pred_c - z_next_c)**2)
        loss_couple = torch.mean((z_t_proj - z_c)**2)

        loss = loss_cycle + 0.5 * loss_time + 0.1 * loss_couple

        # stability
        eigvals = torch.linalg.eigvals(dynamics.K)
        loss += 1e-3 * torch.mean(torch.relu(torch.abs(eigvals) - 1.0))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())

    # --- Validate ---
    dynamics.eval()
    if FREEZE_AE:
        ae.eval()

    val_losses = []

    with torch.no_grad():
        for (batch_time, batch_cycle) in zip(val_time_loader, val_cycle_loader):

            # ===== TIME =====
            x_t_time    = batch_time['x_t'].to(device)
            x_next_time = batch_time['x_next'].to(device)
            u_time      = batch_time['u_t'].to(device)

            z_t      = time_ae.encode(x_t_time)
            z_next_t = time_ae.encode(x_next_time)
            z_pred_t = time_dynamics(z_t, u_time)

            # ===== CYCLE =====
            x_t_cycle    = batch_cycle['x_t'].to(device)
            x_next_cycle = batch_cycle['x_next'].to(device)
            u_cycle      = batch_cycle['u_t'].to(device)

            x_t_sc    = scale(x_t_cycle)
            x_next_sc = scale(x_next_cycle)

            z_c      = ae.encode(x_t_sc)
            z_next_c = ae.encode(x_next_sc)
            z_pred_c = dynamics(z_c, u_cycle)

            # ===== COUPLING =====
            B = x_t_cycle.shape[0]
            z_t_seq = z_t.view(B, -1, z_t.shape[-1])

            z_t_attn = attn(z_t_seq)
            z_t_proj = proj(z_t_attn)

            # ===== LOSSES =====
            loss_time   = torch.mean((z_pred_t - z_next_t)**2)
            loss_cycle  = torch.mean((z_pred_c - z_next_c)**2)
            loss_couple = torch.mean((z_t_proj - z_c)**2)

            loss = loss_cycle + 0.5 * loss_time + 0.1 * loss_couple

            val_losses.append(loss.item())

    tr = np.mean(train_losses)
    va = np.mean(val_losses)

    scheduler.step(va)
    print(f"Epoch {epoch:4d}/{N_EPOCHS} | train {tr:.4e} | val {va:.4e}"
          + (" ← best" if va < best_val_loss else ""))

    # --- Save best checkpoint ---
    if va < best_val_loss:
        best_val_loss = va
        torch.save({
            'epoch':      epoch,
            'dynamics':   dynamics.state_dict(),
            'ae':         ae.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_loss':   best_val_loss,
            'latent_dim': LATENT_DIM,
            'K_matrix':   dynamics.K.detach().cpu(),  # save K explicitly for easy inspection
        }, os.path.join(SAVE_DIR, "best.pth"))

    # --- Checkpoint every 50 epochs ---
    if epoch % 50 == 0:
        torch.save({
            'epoch':    epoch,
            'dynamics': dynamics.state_dict(),
            'val_loss': va,
        }, os.path.join(SAVE_DIR, f"epoch_{epoch}.pth"))

# ── 9. Final summary ──────────────────────────────────────
print(f"\n{'='*50}")
print(f"Training complete.")
print(f"Best val loss : {best_val_loss:.4e}")
print(f"Checkpoints   : {SAVE_DIR}")

# Print learned K matrix
K = dynamics.K.detach().cpu().numpy()
print(f"\nLearned Koopman K matrix ({LATENT_DIM}×{LATENT_DIM}):")
print(np.array2string(K, precision=4, suppress_small=True))

# Eigenvalues — tells you about stability
eigvals = np.linalg.eigvals(K)

print(f"\nEigenvalues of K: {eigvals}")
print(f"All |λ| < 1 (stable): {all(abs(e) < 1 for e in eigvals)}")

print("K norm:", dynamics.K.norm())
print("B norm:", dynamics.B.norm() if dynamics.B is not None else 0)