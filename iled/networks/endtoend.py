from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from typing import Any, Optional, Sequence, Callable, Union


@dataclass
class EndToEndConfig:

    n_warmup: int = 0
    data_dt: float = 1
    substeps: int = 1
    init_nTmax: int = 1
    ae_config: ... = None
    dynamics_config: ... = None

    def make(self):
        return EndToEndModel(self)


class EndToEndModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ae = config.ae_config.make()
        self.dynamics = config.dynamics_config.make()
        self.data_dt = config.data_dt
        self.substeps = config.substeps
        self.n_warmup = config.n_warmup
        assert (
            config.init_nTmax >= config.n_warmup
        ), "NTmax should be at superior to n_warmup"
        self.nTmax = config.init_nTmax
        self._config = config

    # In endtoend.py — replace the forward() method only

    def forward(self, batch):
        """
        batch is a dict with keys:
            'x_t'    : (B, *obs_shape)   — current observation
            'x_next' : (B, *obs_shape)   — next observation
            'u_t'    : (B, control_dim)  — optional control input, or None

        Forward pass:
            z      = encoder(x_t)
            z_next = encoder(x_t_plus_1)          # target in latent space
            z_next_pred = dynamics(z, u_t)         # Koopman prediction
        """
        x_t    = batch['x_t']
        x_next = batch['x_next']
        u_t    = batch.get('u_t', None)           # None if no control

        # Encode both observations independently
        z      = self.ae.encoder(x_t)             # (B, latent_dim)
        z_next = self.ae.encoder(x_next)          # (B, latent_dim)  ← prediction target

        # One-step Koopman prediction
        z_next_pred = self.dynamics(z, u_t)       # (B, latent_dim)

        # Reconstruct x_t for the reconstruction loss
        reconstruction = self.ae.decoder(z)       # (B, *obs_shape)

        # Decode predicted next latent for optional reconstructed-forecast loss
        reconstructed_forecast = self.ae.decoder(z_next_pred)  # (B, *obs_shape)

        linear_part, nl_part = self.dynamics.evaluate_dynamics_parts(z)

        return {
            "z":                     z,
            "z_next":                z_next,           # encoder(x_{t+1}), the target
            "z_next_pred":           z_next_pred,      # K @ z,           the prediction
            "reconstruction":        reconstruction,
            "reconstructed_forecast": reconstructed_forecast,
            # kept for losslib compatibility:
            "true_latents":          z_next.unsqueeze(1),
            "latent_forecast":       z_next_pred.unsqueeze(1),
            "dynamics_parts":        (linear_part, nl_part),
            "memories":              None,
            "additional_losses":     self.dynamics.get_dynamics_losses(),
        }
    def get_nTmax(self): return 1
    def set_nTmax(self, v): pass

    def config(self):
        return self._config

    def decayable_parameters(self):
        list_param = list(self.ae.parameters()) + list(
            self.dynamics.decayable_parameters()
        )

        return iter(list_param)

    def non_decayable_parameters(self):
        return self.dynamics.non_decayable_parameters()

    # def set_nTmax(self, new_nTmax):
    #     self.nTmax = new_nTmax

    # def get_nTmax(self):

    #     return self.nTmax

    def get_n_warmup(self):
        return self.n_warmup

    def get_substeps(self):
        return self.substeps
