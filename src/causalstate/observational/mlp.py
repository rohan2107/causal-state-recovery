import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)

def make_dataloader(
    split: dict[str, np.ndarray],
    scaler: StandardScaler | None = None,
    batch_size: int = 256,
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple[DataLoader, StandardScaler]:

    X = split["obs"].astype(np.float32)
    y = split["Y"].astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1),
    )

    generator = None

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )

    return loader, scaler

def train_mlp(
    split: dict[str, np.ndarray],
    epochs: int = 30,
    lr: float = 1e-3,
    hidden: int = 64,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[MLP, StandardScaler]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, scaler = make_dataloader(
        split,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    model = MLP(
        hidden=hidden,
    )
    fit(
        model,
        train_loader,
        epochs=epochs,
        lr=lr,
    )

    return model, scaler


def fit(
    model: MLP,
    train_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
) -> None:

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    model.train()

    for _ in range(epochs):

        for X, y in train_loader:

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

@torch.no_grad()
def evaluate(
    model: MLP,
    loader: DataLoader,
) -> float:

    model.eval()

    correct = 0
    total = 0

    for X, y in loader:

        logits = model(X)

        pred = (logits >= 0).float()

        correct += (pred == y).sum().item()

        total += len(y)

    return correct / total
