import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "writing" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


def rho_separation():
    with open(RESULTS / "rho_sweep.json") as f:
        data = json.load(f)

    results = data["results"]
    rho = np.array([row["rho"] for row in results])

    sparse = np.array([
        [seed["sparse_gate"]["separation"] for seed in row["seed_results"]]
        for row in results
    ])

    casn = np.array([
        [seed["casn"]["separation"] for seed in row["seed_results"]]
        for row in results
    ])

    sparse_mean = sparse.mean(axis=1)
    sparse_sd = sparse.std(axis=1, ddof=1)
    casn_mean = casn.mean(axis=1)
    casn_sd = casn.std(axis=1, ddof=1)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.errorbar(
        rho, sparse_mean, yerr=sparse_sd,
        marker="o", capsize=3,
        label="Sparse gated selection",
    )

    ax.errorbar(
        rho, casn_mean, yerr=casn_sd,
        marker="s", capsize=3,
        label="CaSN",
    )

    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.axhline(
        0.650,
        linestyle=":",
        linewidth=0.8,
        label="Replay-PN oracle",
    )

    ax.set_xlabel(r"Corruption strength $\rho$")
    ax.set_ylabel(r"Separation $\Delta_{\mathrm{sep}}$")
    ax.set_xticks(rho)
    ax.set_ylim(-0.08, 0.70)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES / "rho_separation.pdf", bbox_inches="tight")
    plt.close(fig)


def downstream_ood():
    with open(RESULTS / "downstream_rq2.json") as f:
        data = json.load(f)

    full = {row["seed"]: row for row in data["results"]["full"]}
    pruned = {row["seed"]: row for row in data["results"]["pruned"]}

    seeds = sorted(full)
    x = np.array([0, 1])

    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    pruned_lines = []
    for seed in seeds:
        line, = ax.plot(
            x,
            [
                pruned[seed]["train_success"],
                pruned[seed]["ood_success"],
            ],
            marker="s",
            linestyle="--",
            linewidth=1.5,
            color="C0",
            zorder=4,
        )
        pruned_lines.append(line)

    full_lines = []
    for seed in seeds:
        line, = ax.plot(
            x,
            [
                full[seed]["train_success"],
                full[seed]["ood_success"],
            ],
            marker="o",
            linewidth=1.2,
            color="C1",
            zorder=3,
        )
        full_lines.append(line)

    ax.annotate(
        "Seeds 0, 1, 4 coincide at 1.00",
        (0, 1.0),
        xytext=(8, -18),
        textcoords="offset points",
        va="top",
    )

    ax.annotate(
        "Seed 3",
        (1, full[3]["ood_success"]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
    )

    ax.annotate(
        "Seed 2",
        (1, full[2]["ood_success"]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "OOD"])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.75, 1.02)

    ax.legend(
        handles=[pruned_lines[0], full_lines[0]],
        labels=["Pruned representation", "Full observation"],
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(FIGURES / "downstream_ood.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rho_separation()
    downstream_ood()
