import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from causalstate.evaluation.report import report
from causalstate.observational.mlp import fit, make_dataloader
from causalstate.world.gridworld import ALL_VARS, VAR_SLICES

GATE_TAU = 0.1
DEFAULT_L1 = 0.02


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 64,
        gate_init: float = 2.0,
    ) -> None:
        super().__init__()

        self.gate_logits = nn.Parameter(
            torch.full((len(ALL_VARS),), gate_init)
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

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _expanded_gate(self) -> Tensor:
        return torch.sigmoid(self.gate_logits)[self.col_to_var]

    def forward(self, x: Tensor) -> Tensor:
        gate = self._expanded_gate()
        return self.net(x * gate)



def gate_penalty(
    model: GatedMLP,
) -> Tensor:
    return torch.sigmoid(
        model.gate_logits
    ).sum()

def train_gate(
    split: dict[str, np.ndarray],
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    hidden: int = 64,
    batch_size: int = 256,
    l1: float = DEFAULT_L1,
    seed: int = 0,
) -> tuple[GatedMLP, StandardScaler]:
    torch.manual_seed(seed)

    train_loader, scaler = make_dataloader(
        split,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    model = GatedMLP(
        hidden=hidden,
    )

    fit(
        model,
        train_loader,
        epochs=epochs,
        lr=lr,
        penalty=lambda m: l1 * gate_penalty(m),
    )

    return model, scaler

def gate_values(
    model: GatedMLP,
) -> dict[str, float]:
    scores = torch.sigmoid(model.gate_logits)
    return {
        var: score.item()
        for var, score in zip(ALL_VARS, scores, strict=True)
    }

def run_sparse_gate(
    splits: dict[str, dict[str, np.ndarray]],
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    hidden: int = 64,
    batch_size: int = 256,
    l1: float = DEFAULT_L1,
    seed: int = 0,
) -> dict:
    model, scaler = train_gate(
        splits["train"],
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        l1=l1,
        seed=seed,
    )

    scores = gate_values(model)

    return report(
        scores,
        model,
        scaler,
        splits,
        threshold=GATE_TAU,
    )
