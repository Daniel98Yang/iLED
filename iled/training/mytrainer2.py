from . import losslib
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from ..backprop import ACA_bptt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
import time
import os
import os.path as osp
from typing import Any, Optional, Sequence, Callable, Union
from math import inf

class TrainingStats:

    def __init__(self):

        self.losses = []
        self.val_losses = []
        self.losses_std = []
        self.val_losses_std = []
        self.grad_norms = []
        self.nTmax_lst = []
        self.best_valid_epoch = 0
        self.best_valid_loss = float("inf")

    def state_dict(self):

        return {
            "losses": self.losses,
            "val_losses": self.val_losses,
            "losses_std": self.losses_std,
            "val_losses_std": self.val_losses_std,
            "grad_norms": self.grad_norms,
            "nTmax_lst": self.nTmax_lst,
            "best_valid_epoch": self.best_valid_epoch,
            "best_valid_loss": self.best_valid_loss,
        }

    def load_state_dict(self, state_dict):

        for k in state_dict.keys():
            if k in self.__dict__.keys():
                self.__dict__[k] = state_dict[k]


@dataclass
class TrainerConfig:
    """Specification of optimizer and scheduler."""

    model_config: ...
    save_path: str
    losses_and_scales: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )

    dtype: torch.dtype = torch.float

    optimizer: str = "adam"
    optimizer_kwargs: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    lr: float = 0.001
    weight_decay: float = 0.0

    max_epochs: int = 999_999_999  # Early stopping criteria, max num epochs
    max_patience: int = (
        0  # Early stopping criteria, max num epochs without improvements
    )
    t_increment_patience: int = 10  # Length increase patience for end to end trainers
    target_length: int = 0  # Maximum sequence length
    nT_increment: int = 1

    cuda: bool = False

    parallel: bool = False
    validate_every: int = 1
    checkpoint_every: int = 10

    scaler_config: ... = None

    lr_reduction_order: float = 0.
    loss_threshold: float = float(inf)

    def make(self, model=None):

        return Trainer(self, model=model)


optimizer_map = {"adam": torch.optim.Adam}


class Trainer:

    def __init__(self, config: TrainerConfig, model=None):
        loss = losslib.KoopmanLoss(recon_scale=1.0, latent_scale=1.0, forecast_scale=0.5)

        self.config = config
        if model is None:
            self.model = config.model_config.make()
        else:
            self.model = model
        self.model = self.model.to(config.dtype)

        if config.cuda:
            self.model.cuda()

        self.loss = loss

        self.optimizer = optimizer_map[config.optimizer](
            self.model.decayable_parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs,
        )

        new_group_kwargs = {
            "params": self.model.non_decayable_parameters(),
            "lr": self.config.lr,
            "weight_decay": 0.0,
        }
        new_group_kwargs.update(config.optimizer_kwargs)
        self.optimizer.add_param_group(new_group_kwargs)

        self.cur_epoch = 0

        self.save_path = config.save_path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        if config.parallel:
            self.model = nn.DataParallel(self.model)

        self.stats = TrainingStats()

        if config.scaler_config is not None:
            self.scaler = config.scaler_config.make().to(config.dtype)
            if config.cuda:
                self.scaler.cuda()
        else:
            self.scaler = None

        if config.lr_reduction_order > 0.:
            
            self.reduction_rate = np.exp(np.log(10**(-config.lr_reduction_order))/(config.target_length/config.nT_increment))
        else: self.reduction_rate = 1

    def load_state_dict(self, state: dict):
        self.optimizer.load_state_dict(state["optimizer"])
        self.model.load_state_dict(state["model"])
        self.stats.load_state_dict(state["stats"])

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "stats": self.stats.state_dict(),
        }

    def get_hyperparams(self) -> dict:
        """Return learning rates of each parameter group (usually only one)."""
        if len(self.optimizer.param_groups) == 1:
            return {"lr": [group["lr"] for group in self.optimizer.param_groups][0]}
        return {"lr": [group["lr"] for group in self.optimizer.param_groups]}

    def train(self, train_loader, valid_loader=None):

        self.cur_epoch = len(self.stats.losses)

        best_valid_loss = self.stats.best_valid_loss
        best_valid_epoch = self.stats.best_valid_epoch

        while True:
            if self.should_stop():
                break

            # Train epoch
            out = self.evaluate_dataset(train_loader, train=True)
            self.stats.losses.append(out["loss_mean"])
            self.stats.losses_std.append(out["loss_std"])
            self.stats.nTmax_lst.append(self.model.get_nTmax())

            # perform validation
            improved = False
            if self.cur_epoch % self.config.validate_every == 0:
                out = self.evaluate_dataset(valid_loader, train=False)

                if out["loss_mean"] < best_valid_loss:
                    best_valid_loss = out["loss_mean"]
                    best_valid_epoch = self.cur_epoch
                    improved = True
                self.stats.val_losses.append(out["loss_mean"])
                self.stats.val_losses_std.append(out["loss_std"])
                self.stats.best_valid_epoch = best_valid_epoch
                self.stats.best_valid_loss = best_valid_loss

                no_imp = self.cur_epoch - best_valid_epoch

                cur_nTmax = self.model.get_nTmax()

                if (
                    no_imp >= self.config.t_increment_patience
                    and self.config.target_length > cur_nTmax
                    and self.stats.losses[-1]<self.config.loss_threshold
                    ):
                    increment = min(
                        self.config.target_length - cur_nTmax, self.config.nT_increment
                    )
                    self.model.set_nTmax(self.model.get_nTmax() + increment)
                    best_valid_loss = float("inf")
                    self.reduce_lr()

            self.cur_epoch += 1

            # if eval is improved, save net
            if improved:
                # save model here
                self.save(best=True)

            if self.cur_epoch % self.config.checkpoint_every == 0:
                self.save(best=False)

    def should_stop(self) -> bool:
        if self.cur_epoch >= self.config.max_epochs:
            return True

        try:
            no_imp = self.cur_epoch - self.stats.best_valid_epoch
        except IndexError:
            no_imp = 0

        if (0 < self.config.max_patience <= no_imp
            and self.stats.losses[-1]<self.config.loss_threshold):
            return True

        return False

    def evaluate_dataset(self, data_loader, train=True):
        losses = []
        n = 0
        was_training = self.model.training
        if train:
            self.model.train()
            desc = f"Epoch {self.cur_epoch} | Train | "
        else:
            self.model.eval()
            desc = f"Epoch {self.cur_epoch} |  Val  | "
        pbar = tqdm(data_loader)

        for input_batch in pbar:
            self.optimizer.zero_grad()
            input_batch = input_batch.to(self.config.dtype)
            if self.config.cuda:
                input_batch = input_batch.cuda()

            if self.scaler:
                input_batch = self.scaler.scaling(input_batch)

            output = self.model(input_batch)
            itr_loss = self.loss(output, input_batch)
            if train:
                itr_loss.backward()
                self.optimizer.step()

            losses.append(itr_loss.item())

            pbar.set_description(
                desc
                + "nTmax : {}, {:.3e}".format(
                    self.model.get_nTmax(), np.array(losses).mean()
                )
            )
        out = {"loss_mean": np.array(losses).mean(), "loss_std": np.array(losses).std()}
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        return out

    def reduce_lr(self):

        for g in self.optimizer.param_groups:
            g['lr'] = self.reduction_rate*g['lr']

    def save(self, best=False):
        torch.save(self.config, osp.join(self.save_path, "config.trch"))
        state = self.state_dict()

        if best:
            torch.save(state, osp.join(self.save_path, "best.trch"))
        else:
            torch.save(state, osp.join(self.save_path, "last.trch"))

    def load(self, name):
        """
        Loads the trainer state (including model weights) from a given checkpointing
        file.
        name (str): name of the file to be loaded
        """
        load_path = osp.join(self.save_path, name)
        if self.config.cuda:
            dic = torch.load(load_path, map_location=torch.device("cuda"))
        else:
            dic = torch.load(load_path, map_location=torch.device("cpu"))

        if self.config.parallel:
            self.model.module.load_state_dict(dic["model"])
        else:
            self.model.load_state_dict(dic["model"])

        self.optimizer.load_state_dict(dic["optimizer"])
        self.stats.load_state_dict(dic["stats"])
        return dic
