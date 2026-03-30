import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from koopmandataset  import KoopmanDataset
from koopmandynamics import KoopmanDynamics
from losslib         import KoopmanLoss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ CONFIGURE THESE PATHS AND SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_PATH        = "my_data.npy"          # ★ (N, T, obs_dim) numpy array
CONTROLS_PATH    = None                   # ★ "my_controls.npy" or None if no controls
AUTOENCODER_PATH = "my_autoencoder.pth"   # ★ path to your .pth or .pt file
SAVE_DIR         = "./koopman_checkpoints"

LATENT_DIM  = 3      # ★ must match your autoencoder's latent output
BATCH_SIZE  = 64
LR          = 1e-3
N_EPOCHS    = 500
VAL_SPLIT   = 0.1
USE_CUDA    = torch.cuda.is_available()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if USE_CUDA else "cpu")

# ── 1. Load data ──────────────────────────────────────────────────
trajs    = np.load(DATA_PATH)                                      # (N, T, obs_dim)
controls = np.load(CONTROLS_PATH) if CONTROLS_PATH else None       # (N, T-1, ctrl_dim) or None

OBS_DIM     = trajs.shape[-1]
CONTROL_DIM = controls.shape[-1] if controls is not None else 0

dataset = KoopmanDataset(trajs, controls=controls)
n_val   = int(VAL_SPLIT * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ── 2. Load YOUR autoencoder from .pth / .pt ─────────────────────
# OPTION A: your file contains a state_dict (most common)
#   Define the class inline or import it, then load weights:
#
#   class MyEncoder(nn.Module): ...
#   class MyDecoder(nn.Module): ...
#   class MyAutoEncoder(nn.Module):
#       def __init__(self): ...
#       def forward(self, x): ...
#
# ★ Replace the block below with whichever option matches your file:

# --- OPTION A: state_dict ---
from my_autoencoder import MyAutoEncoder                  # ★ create this file
ae = MyAutoEncoder(input_dim=OBS_DIM, latent_dim=LATENT_DIM)
ae.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=device))

# --- OPTION B: full saved model (uncomment if your .pth is a full model) ---
# ae = torch.load(AUTOENCODER_PATH, map_location=device)

ae = ae.to(device)

# ── 3. Build Koopman dynamics ─────────────────────────────────────
dynamics = KoopmanDynamics(latent_dim=LATENT_DIM, control_dim=CONTROL_DIM).to(device)

# ── 4. Build EndToEndModel cleanly (no __new__ hack) ─────────────
from endtoend import EndToEndModel

class KoopmanEndToEnd(nn.Module):
    """Thin wrapper — avoids needing EndToEndConfig entirely."""
    def __init__(self, ae, dynamics):
        super().__init__()
        self.ae       = ae
        self.dynamics = dynamics

    def forward(self, batch):
        x_t    = batch['x_t'].to(device)
        x_next = batch['x_next'].to(device)
        u_t    = batch['u_t'].to(device) if 'u_t' in batch else None

        z           = self.ae.encoder(x_t)
        z_next      = self.ae.encoder(x_next)
        z_next_pred = self.dynamics(z, u_t)
        reconstruction        = self.ae.decoder(z)
        reconstructed_forecast = self.ae.decoder(z_next_pred)
        linear_part, nl_part  = self.dynamics.evaluate_dynamics_parts(z)

        return {
            "z":                      z,
            "z_next":                 z_next,
            "z_next_pred":            z_next_pred,
            "reconstruction":         reconstruction,
            "reconstructed_forecast": reconstructed_forecast,
            "dynamics_parts":         (linear_part, nl_part),
        }

    def get_nTmax(self): return 1
    def set_nTmax(self, v): pass

model = KoopmanEndToEnd(ae, dynamics)

# ── 5. Optimizer and loss ─────────────────────────────────────────
optimizer = torch.optim.Adam(
    list(ae.parameters()) + list(dynamics.parameters()), lr=LR
)
loss_fn = KoopmanLoss(recon_scale=1.0, latent_scale=1.0, forecast_scale=0.5)

# ── 6. Training loop with logging + checkpointing ────────────────
best_val_loss = float('inf')

for epoch in range(1, N_EPOCHS + 1):

    # --- train ---
    model.train()
    train_losses = []
    for batch in train_loader:
        optimizer.zero_grad()
        out  = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # --- validate ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            out  = model(batch)
            loss = loss_fn(out, batch)
            val_losses.append(loss.item())

    train_mean = np.mean(train_losses)
    val_mean   = np.mean(val_losses)

    print(f"Epoch {epoch:4d}/{N_EPOCHS} | train {train_mean:.4e} | val {val_mean:.4e}")

    # --- save best ---
    if val_mean < best_val_loss:
        best_val_loss = val_mean
        torch.save({
            'epoch':      epoch,
            'ae':         ae.state_dict(),
            'dynamics':   dynamics.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_loss':   best_val_loss,
        }, os.path.join(SAVE_DIR, "best.pth"))

    # --- checkpoint every 50 epochs ---
    if epoch % 50 == 0:
        torch.save({
            'epoch':     epoch,
            'ae':        ae.state_dict(),
            'dynamics':  dynamics.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(SAVE_DIR, f"epoch_{epoch}.pth"))

print(f"\nDone. Best val loss: {best_val_loss:.4e}")
print(f"Checkpoints saved to: {SAVE_DIR}")