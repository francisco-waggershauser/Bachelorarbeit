# EF1/EFX Round-Robin Simulation

Simulation code for the bachelor thesis *The Gap Between EF1 and EFX:
A Simulation-Based Average-Case Analysis of Round-Robin*.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# sanity checks (Tests C, D, E, F)
.venv/bin/python test_sanity.py

# example cells + three figure types
.venv/bin/python main.py
```

Edit grids and `R` at the top of `main.py`. Results go to `results/` (one `.npz` per cell), figures to `figures/`.

## Modules

| File | Role |
|---|---|
| `sampling.py` | `draw_base` → `build_Z` → `build_values` |
| `mechanisms.py` | Plug-in slot `mechanism(V, rng) -> A`; v1 has only `round_robin` |
| `evaluate.py` | Uncapped `alpha` matrix, `alpha_star`, EF1/EFX checks |
| `simulate.py` | `run_cell`, `.npz` I/O, `reconstruct` from `(master_seed, k)` |
| `analyse.py` | Survival, EFX rate, conditional stats, `describe_cell` |
| `scales.py` | Gini(θ) and Spearman(ρ) for plot labels only |
| `plot.py` | Survival panels, rate vs m, rate vs collision |
| `test_sanity.py` | Tests C, D, E (RR-only) and F |
| `main.py` | Example grids and figures |

## Notes

- Cell parameters `n`, `m`, `rho`, `theta` are free (with `n >= 2`, `m > n`, `rho in [0,1]`, `theta > 0`).
- Filenames use `rho{rho:.2f}_theta{theta:.2f}` via `cell_filename` only.
- Plot legends report Gini and Spearman; files stay in `(rho, theta)`.
- Fairness tolerance `eps` is an analysis argument only (default `0.0`).
