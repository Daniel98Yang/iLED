# my_trainscript.py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from my_autoencoder import MyAutoEncoder
from my_dataset import MyDataset

# Import iLED pieces (adjust paths to match what you see in iled/__init__.py)
from iled.nn.endtoend import EndToEndConfig, EndToEndModel
from iled.nn.trainer import TrainerConfig

# ── 1. Load YOUR data ──────────────────────────────────────────────────────────
data = np.load("my_data.npy")          # shape: (N, T, feature_dim)
INPUT_DIM   = data.shape[-1]           # e.g. 50
LATENT_DIM  = 3                        # your 3 latent variables
SEQ_LEN     = data.shape[1]            # e.g. 100

dataset = MyDataset(data)
n_val   = int(0.1 * len(dataset))
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

# ── 2. Build YOUR autoencoder ──────────────────────────────────────────────────
my_ae = MyAutoEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)

# ── 3. Build the EndToEnd model, injecting your AE directly ───────────────────
#    EndToEndConfig.make() calls ae_config.make(), so we bypass that
#    by building EndToEndModel manually and swapping the ae attribute.

from iled.nn.splitdynamics import ...   # import whatever dynamics config the FHN script uses
                                         # check trainscript.py for the exact class name

dynamics_config = ...   # copy from examples/FHN/trainscript.py, but set latent_dim=3

e2e_config = EndToEndConfig(
    n_warmup=10,
    data_dt=1.0,
    substeps=1,
    init_nTmax=11,
    ae_config=None,           # we'll set ae manually below
    dynamics_config=dynamics_config,
)
model = EndToEndModel(e2e_config)
model.ae = my_ae              # ← swap in YOUR autoencoder here

# ── 4. Train ───────────────────────────────────────────────────────────────────
trainer_config = TrainerConfig(
    model_config=None,
    save_path="./my_run",
    losses_and_scales={
        "reconstruction":    ["mse", 1.0],
        "latent_forecast":   ["mse", 1.0],
        "reconstructed_forecast": ["mse", 0.5],
    },
    lr=1e-3,
    max_epochs=500,
    t_increment_patience=20,
    target_length=SEQ_LEN,
    nT_increment=5,
    cuda=torch.cuda.is_available(),
)
trainer = trainer_config.make(model=model)
trainer.train(train_loader, val_loader)