from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from causalstate.evaluation.report import report
from causalstate.observational.mlp import make_dataloader
from causalstate.observational.sparse_gate import (
    GATE_TAU,
    gate_penalty,
)
from causalstate.world.gridworld import ALL_VARS, VAR_SLICES

CASN_DEFAULTS = {
    "epochs": 40,
    "lr": 1e-3,
    "hidden": 64,
    "rep_dim": 32,
    "batch_size": 256,
    "l1": 0.02,
    "target_lambda": 0.1,
    "int_lambda": 0.001,
    "int_reg": 0.01,
    "seed": 0,
    "bias": 1.0,
}


class CaSN(nn.Module):

    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 64,
        rep_dim: int = 32,
        gate_init: float = 2.0,
    ) -> None:
        super().__init__()

        self.gate_logits = nn.Parameter(
            torch.full(
                (len(ALL_VARS),),
                gate_init,
            )
        )

        col_to_var = [0] * in_dim

        for i, var in enumerate(ALL_VARS):
            sl = VAR_SLICES[var]

            for column in range(sl.start, sl.stop):
                col_to_var[column] = i

        self.register_buffer(
            "col_to_var",
            torch.tensor(
                col_to_var,
                dtype=torch.long,
            ),
        )

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, rep_dim),
            nn.ReLU(),
        )

        self.intervener = nn.Linear(
            rep_dim,
            rep_dim,
        )

        self.classifier = nn.Linear(
            rep_dim,
            1,
        )

    def gates(self) -> Tensor:

        return torch.sigmoid(self.gate_logits)

    def _expanded_gate(self) -> Tensor:

        return self.gates()[self.col_to_var]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gates()[self.col_to_var]
        return self.encoder(g * x)

    def forward_all(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        z = self.encode(x)

        delta = self.intervener(z)
        int_z = z + delta

        y = self.classifier(z)
        int_y = self.classifier(int_z)

        return y, int_y, delta, z

    def forward(self, x: Tensor) -> Tensor:

        return self.classifier(
            self.encode(x)
        )

    def gate_values(self) -> dict[str, float]:

        with torch.no_grad():
            values = self.gates().tolist()

        return {
            var: float(value)
            for var, value in zip(
                ALL_VARS,
                values,
                strict=True,
            )
        }

def necessity_loss(
    y_logits: Tensor,
    int_logits: Tensor,
) -> Tensor:

    return -F.mse_loss(
        torch.sigmoid(y_logits),
        torch.sigmoid(int_logits),
    )


def intervention_regularization(
    delta: Tensor,
    bias: float = 1.0,
) -> Tensor:

    return (delta.square() - bias).norm(dim=1).mean()


def sufficiency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        target.float().unsqueeze(1),
    )

def intervened_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return -F.binary_cross_entropy_with_logits(
        logits,
        target.float().unsqueeze(1),
    )

def fit_casn(
    model: CaSN,
    train_loader,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    l1: float = 0.02,
    target_lambda: float = 0.1,
    int_lambda: float = 0.001,
    int_reg: float = 0.01,
    bias: float = 1.0,
) -> None:

    opt_adv = torch.optim.Adam(
        model.intervener.parameters(),
        lr=lr,
    )

    opt_main = torch.optim.Adam(
        [
            model.gate_logits,
            *model.encoder.parameters(),
            *model.classifier.parameters(),
        ],
        lr=lr,
    )

    for _ in range(epochs):
        for X, y in train_loader:
            y = y.squeeze(1)
            opt_adv.zero_grad()

            y_logits, int_logits, delta, _ = model.forward_all(X)

            targets = necessity_loss(
                y_logits,
                int_logits,
            )

            int_nll = intervened_loss(
                int_logits,
                y,
            )

            intervention = intervention_regularization(
                delta,
                bias=bias,
            )

            adv_loss = (
                -target_lambda * targets
                -int_lambda * int_nll
                + int_reg * intervention
            )

            adv_loss.backward()
            opt_adv.step()
            opt_main.zero_grad()

            y_logits, int_logits, delta, _ = model.forward_all(X)

            nll = sufficiency_loss(
                y_logits,
                y,
            )

            targets = necessity_loss(
                y_logits,
                int_logits,
            )

            int_nll = intervened_loss(
                int_logits,
                y,
            )

            main_loss = (
                nll
                + target_lambda * targets
                + int_lambda * int_nll
                + int_reg * intervention_regularization(
                    delta,
                    bias=bias,
                )
                + l1 * gate_penalty(model)
            )

            main_loss.backward()
            opt_main.step()

def train_casn(
    split: dict[str, np.ndarray],
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    hidden: int = 64,
    rep_dim: int = 32,
    batch_size: int = 256,
    l1: float = 0.02,
    target_lambda: float = 0.1,
    int_lambda: float = 0.001,
    int_reg: float = 0.01,
    bias: float = 1.0,
    seed: int = 0,
) -> tuple[CaSN, StandardScaler]:

    torch.manual_seed(seed)

    train_loader, scaler = make_dataloader(
        split,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    model = CaSN(
        hidden=hidden,
        rep_dim=rep_dim,
    )

    fit_casn(
        model,
        train_loader,
        epochs=epochs,
        lr=lr,
        l1=l1,
        target_lambda=target_lambda,
        int_lambda=int_lambda,
        int_reg=int_reg,
        bias=bias,
    )

    return model, scaler

@torch.no_grad()
def intervention_magnitude(
    model: CaSN,
    loader,
) -> float:

    model.eval()

    total = 0.0
    count = 0

    for X, _ in loader:
        _, _, delta, _ = model.forward_all(X)

        magnitude = delta.square().sum(dim=1)

        total += magnitude.sum().item()
        count += len(X)

    return total / count

def run_casn(
    splits: dict[str, dict[str, np.ndarray]],
    **kwargs,
) -> dict:
    model, scaler = train_casn(
        splits["train"],
        **kwargs,
    )

    scores = model.gate_values()

    return report(
        scores,
        model,
        scaler,
        splits,
        threshold=GATE_TAU,
    )
