# causal-state-recovery

Recovering causal state variables for reinforcement learning via probabilities of necessity and
sufficiency.

MSc Data Science dissertation, University of Bristol (EMATM0047).

A MiniGrid `DoorKey-6x6` environment with planted causal ground truth: four state variables drive
the dynamics, one is pure noise, and one is a corrupted copy of a real cause. Observational
selection methods can therefore be scored against a known answer, while a replay-based
interventional oracle measures each variable's probability of necessity directly.

## The state variables

| slice | var | contents | causal |
|---|---|---|---|
| `0:3` | `s0` | agent pose — column, row, direction | yes |
| `3:4` | `s1` | agent is carrying the key | yes |
| `4:5` | `s2` | door is open | yes |
| `5:7` | `s3` | goal position — column, row | yes |
| `7:8` | `s4` | Gaussian noise | no |
| `8:9` | `s5` | copy of `s2`, inverted with probability `1 - rho` | no |

`s5` is the spurious channel: at `rho = 1` it equals `s2` exactly, at `rho = 0` it is exactly
inverted, and at no `rho` is it read by the dynamics.

## Usage

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Experiments live in [scripts/](scripts/). Train a policy first, then run any experiment script;
each writes a single JSON file to `data/results/`.

```bash
python scripts/train_policy.py --seed 0
python scripts/oracle_recovery.py
```

`data/` and `models/` are gitignored by design — the scripts are the reproducible artifact, not
their outputs.

## Licence

[MIT](LICENSE).
