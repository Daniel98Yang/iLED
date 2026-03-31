# koopman_dynamics.py
import torch
import torch.nn as nn

class KoopmanDynamics(nn.Module):
    def __init__(self, latent_dim, control_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        # 1. DEFINE K
        self.K = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.eye_(self.K)
        if control_dim > 0:
            self.B = nn.Parameter(torch.randn(latent_dim, control_dim) * 0.01)
        else:
            self.B = None

    def forward(self, z, u=None):
        """Single-step prediction. z: (B, latent_dim), u: (B, control_dim) or None"""
        z_next = z @ self.K.T
        if self.B is not None and u is not None:
            z_next = z_next + u @ self.B.T
        return z_next

    def integrate(self, z_seq, u_seq=None):
        """
        Roll out K over a full sequence for inference/plotting.
        z_seq: (B, T, latent_dim)  — only z_seq[:,0,:] is used as IC
        u_seq: (B, T-1, control_dim) or None
        Returns: (B, T, latent_dim) predicted trajectory
        """
        B, T, _ = z_seq.shape
        preds = [z_seq[:, 0, :]]
        for t in range(T - 1):
            u = u_seq[:, t, :] if u_seq is not None else None
            preds.append(self.forward(preds[-1], u))
        return torch.stack(preds, dim=1)   # (B, T, latent_dim)

    def evaluate_dynamics_parts(self, z, memories=None):
        """
        Returns the linear part (K@z) and None for nonlinear part.
        Matches the (linear_part, nl_part) tuple that losslib expects.
        """
        linear_part = z @ self.K.T
        return linear_part, None            # nl_part = None → nl_penalisation skipped

    def get_dynamics_losses(self):
        return [None]                       # no auxiliary losses

    def decayable_parameters(self):
        return iter([])                     # K is a weight matrix — don't decay it

    def non_decayable_parameters(self):
        params = [self.K]
        if self.B is not None:
            params.append(self.B)
        return iter(params)