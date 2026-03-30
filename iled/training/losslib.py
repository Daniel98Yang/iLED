import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Union
import copy


def centering_loss(array, *args, **kwargs):
    return torch.mean(array.mean(-2, keepdim=True) ** 2)

class KoopmanLoss:
    """
    Clean Koopman loss that only uses what the new forward pass produces.
    """
    def __init__(self, recon_scale=1.0, latent_scale=1.0, forecast_scale=0.0):
        self.recon_scale    = recon_scale     # weight for reconstruction loss
        self.latent_scale   = latent_scale    # weight for latent prediction loss ← your main loss
        self.forecast_scale = forecast_scale  # weight for decoded forecast loss (optional)

    def __call__(self, output_dict, batch):
        z_next      = output_dict["z_next"]             # encoder(x_{t+1})
        z_next_pred = output_dict["z_next_pred"]        # K @ encoder(x_t)
        recon       = output_dict["reconstruction"]     # decoder(encoder(x_t))
        recon_next  = output_dict["reconstructed_forecast"]

        x_t    = batch["x_t"]
        x_next = batch["x_next"]

        # Your requested loss: latent one-step prediction
        loss_latent = ((z_next_pred - z_next) ** 2).mean()

        # Reconstruction loss: encoder→decoder should be identity
        loss_recon = ((recon - x_t) ** 2).mean()

        loss = self.latent_scale * loss_latent + self.recon_scale * loss_recon

        if self.forecast_scale > 0:
            loss_forecast = ((recon_next - x_next) ** 2).mean()
            loss += self.forecast_scale * loss_forecast

        return loss

class ScaledLosses:
    def __init__(self, parent_loss: Callable, scale: float):
        self.parent = parent_loss
        self.scale = scale

    def __repr__(self):
        return f"{self.__class__.__name__}(parent={self.parent!r}, scale={self.scale})"

    def __call__(self, *args, **kwargs):
        return self.scale * self.parent(*args, **kwargs)


loss_map = {"mse": F.mse_loss, "l1": F.l1_loss, "centering_loss": centering_loss,
            'norm_loss':torch.norm}


class EndToEndLosses:

    def __init__(self, losses_and_scales={}):
        self.losses_and_scales = {}

        for k in losses_and_scales.keys():
            self.losses_and_scales[k] = [loss_map[losses_and_scales[k][0]]]
            self.losses_and_scales[k].append(losses_and_scales[k][1])

        self.reconstruction_loss = ScaledLosses(
            *self.losses_and_scales["reconstruction"]
        )
        self.latent_forecast_loss = ScaledLosses(
            *self.losses_and_scales["latent_forecast"]
        )

        if "reconstructed_forecast" in self.losses_and_scales.keys():
            self.reconstructed_forecast_loss = ScaledLosses(
                *self.losses_and_scales["reconstructed_forecast"]
            )
        else:
            self.reconstructed_forecast_loss = None

        if "nl_penalisation" in self.losses_and_scales.keys():
            self.nl_penalisation_loss = ScaledLosses(
                *self.losses_and_scales["nl_penalisation"]
            )
        else:
            self.nl_penalisation_loss = None

        if "latent_center" in self.losses_and_scales.keys():
            self.latent_centering_loss = ScaledLosses(
                *self.losses_and_scales["latent_center"]
            )
        else:
            self.latent_centering_loss = None

    def __repr__(self):
        return f"{self.__class__.__name__}(losses_and_scales={self.losses_and_scales})"

    def __call__(self, output_dict, targets_batch):
        targets = targets_batch
        true_latents = output_dict["true_latents"]
        latent_forecast = output_dict["latent_forecast"]
        reconstruction = output_dict["reconstruction"]
        reconstructed_forecast = output_dict["reconstructed_forecast"]
        linear_part, nl_part = output_dict["dynamics_parts"]

        T = latent_forecast.shape[1]
        if linear_part is not None:
            n_warmup = T-linear_part.shape[1]
        else: n_warmup = 0


        loss = self.reconstruction_loss(reconstruction, targets)
        loss += self.latent_forecast_loss(latent_forecast[:,n_warmup:], true_latents[:,n_warmup:T])
        if self.reconstructed_forecast_loss:
            loss += self.reconstructed_forecast_loss(
                reconstructed_forecast, targets[:, :T]
            )

        if self.nl_penalisation_loss and nl_part is not None:
            loss += self.nl_penalisation_loss(nl_part)

        if self.latent_centering_loss:
            loss += self.latent_centering_loss(true_latents)

        if "additional_losses" in output_dict.keys():
            # expects scalar additional losses
            for l in output_dict["additional_losses"]:
                loss = loss + l.mean() if l is not None else loss

        return loss
