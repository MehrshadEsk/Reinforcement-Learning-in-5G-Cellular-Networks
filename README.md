# QoS-aware CIO control in mobile cellular networks

An auditable research implementation accompanying **“Deep Reinforcement Learning Approach to QoS Aware Load Balancing in 5G Cellular Networks under User Mobility and Observation Uncertainty”**, manuscript by M. Eskandarpour and H. Soleimani.

**Status: a newly written, manuscript-grounded reconstruction.** The uploaded PDF did not contain the original code, training logs, checkpoints, random seeds, or complete simulator parameters. This package implements the described research workflow with explicitly documented choices. Its outputs are new experiments, not recovered evidence for the manuscript's reported results. Read [the manuscript audit](supplement/MANUSCRIPT_AUDIT.md) before using these files in a submission.

## Start here

Use Python 3.10 or newer. Python 3.12 was used for the included verification.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python -m qos_cio demo --out runs/my_demo
```

The package runs on CPU and does not require a GPU, ns-3, MATLAB, or proprietary data. For a smaller CPU-only PyTorch installation, install `torch` using its official CPU package index before installing this package. The `requirements-verified.txt` file records the numerical packages actually used here. Use the version ranges in `pyproject.toml` for other supported platforms; cross-version bitwise equivalence is not promised.

The demo deliberately uses 12 UEs, four 8-second training episodes per seed, two training seeds, and two held-out environment seeds. It proves that the workflow executes; it is too short to demonstrate policy convergence or a reliable algorithm ranking. Included outputs are under `runs/demo/`.

Output directories must be new. Commands refuse to silently overwrite an existing run.

## What is included

| Location | Purpose |
|---|---|
| `qos_cio/environment.py` | Radio propagation, mobility, queues, scheduling, bounded CIO, handover, telemetry, packet and mobility accounting |
| `qos_cio/agents.py` | Squashed-Gaussian PPO, GAE integration, twin-Q discrete adaptation, checkpoints |
| `qos_cio/experiments.py` | Training, frozen evaluation, paired seeds, confidence intervals, reward reference calibration |
| `qos_cio/studies.py` | Density, mobility, noise, delay, missingness, timing, resolution and retrained reward ablations; simplex weight search |
| `qos_cio/plots.py` | Vector/PDF and PNG figures generated from actual logs, not prescribed trends |
| `configs/` | Explicit defaults, literal/augmented states, demo, ablations, disjoint-seed manifest |
| `tests/` | Numerical, accounting, policy-distribution, handover and reproducibility checks |
| `supplement/` | Deep manuscript audit, parameter provenance, mathematical supplement, reproduction protocol, result dictionary and figure map |
| `reported/` | Manually transcribed manuscript Tables 4–5, clearly separated from simulation output |
| `scripts/` | Full-study commands, checkpoint evaluation, figure generation and archive verification |
| `analysis.ipynb` | Local notebook for reading results and reproducing plots |
| `runs/demo/` | Executed smoke-scale training/evaluation, checkpoints, data, logs, figures and provenance |

## Important modeling decisions

- **Radio:** full-frequency-reuse downlink interference, distance path loss, correlated shadowing, Rayleigh fading, one effective spatial layer, 106 × 180 kHz PRBs. CQI is an explicitly documented monotone proxy. The simulator is not a calibrated NR PHY.
- **Queues and delay:** finite FIFO waiting queues, fixed packet size, four abstract stop-and-wait processes, probabilistic decoding failure and capped retries. No full RLC-AM segmentation/reordering, HARQ soft combining, MCS/BLER curves, core-network transport, or 3GPP RLF state machine is claimed.
- **Time:** mobility updates every 100 ms; packet service advances on a separate 10 ms tick; CIO changes every second. Short-latency conclusions require the provided finer-tick sensitivity study.
- **State:** `default.yaml` has the six stated KPI channels plus RSRP/CQI (24 features for three cells), resolving the prose's radio-input requirement. `paper6.yaml` preserves Eqs. 34/50 literally (18 features). `radio_cio.yaml` additionally exposes current CIO (27 features), as a declared extension.
- **Baselines:** A3 shares radio filtering, hysteresis and TTT. ReBuHa is an explicitly adapted threshold offloader. CDQL is a discrete twin-Q **incremental-action adaptation**, not an assertion that reference [66]'s original simulator or code has been recovered.
- **Reward:** Eq. 38 remains the reported QoS reward. PPO's 0.10 squared-action regularizer is a separate differentiable optimization term, so logged QoS reward is not silently changed. Metric names prevent the different weight orders in the manuscript from being interchanged.
- **Statistics:** learned policies are frozen. Test traces are shared across algorithms. Trace averages are formed within training seed before computing a Student-t interval across training seeds. One seed yields “not estimable,” never a fabricated zero-width interval.

## Research workflow

The defaults match the manuscript where specified and use explicit assumptions elsewhere. Review `supplement/ASSUMPTIONS.md` and obtain the unresolved author values first.

1. Run correctness tests and the short demonstration.
2. Freeze simulator choices and the primary estimands.
3. Calibrate reward bounds on a disjoint calibration split:

```bash
python -m qos_cio calibrate --config configs/default.yaml \
  --seeds 10000:10010 --out runs/calibration
```

4. Run the ten-seed main experiment:

```bash
python -m qos_cio suite --config runs/calibration/calibrated.yaml \
  --training-seeds 0:10 --eval-seeds 30000:30010 --out runs/main
```

This requests **600 episodes × 600 control steps × 10 seeds × 2 learned algorithms**, before evaluation or ablations. It is a substantial experiment, not a quick demo. No wall-time estimate should be extrapolated without benchmarking the chosen machine and configuration.

5. Evaluate frozen policies under perturbations:

```bash
python -m qos_cio study --config runs/calibration/calibrated.yaml \
  --kind density --checkpoints runs/main/checkpoints.json \
  --eval-seeds 30000:30010 --out runs/density
```

Other kinds: `noise`, `delay`, `missing`, `speed`, `mobility_alpha`, `control`, `resolution`. These are separate one-factor-at-a-time studies; `noise` uses coupled low/medium/high RSRP–CQI–delay triplets. They are not an exhaustive factorial design.

6. Retrain each reward ablation with equal budgets:

```bash
python -m qos_cio study --config runs/calibration/calibrated.yaml \
  --kind ablation --training-seeds 0:10 --eval-seeds 30000:30010 \
  --out runs/ablations
```

7. Optionally perform the specified weight-selection procedure, using the validation split only:

```bash
python -m qos_cio weight-search --config runs/calibration/calibrated.yaml \
  --grid-step 0.25 --training-seeds 0:10 --validation-seeds 20000:20010 \
  --out runs/weight_search
```

The coarse grid is an explicit implementation choice: 0.25 yields 126 candidates and does **not** contain the manuscript's 0.35/0.20/0.20/0.10/0.10/0.05 vector. A 0.05 grid contains that vector but is much larger. The code never pretends that the Table 1 weights were recovered by a search that was not executed. A candidate must satisfy mean latency ≤25 ms and PLR ≤5%; if none qualifies, the selection result says so.

## Individual commands

```bash
python -m qos_cio train --config configs/default.yaml --method ppo --seed 0 --out runs/ppo_0
python -m qos_cio train --config configs/default.yaml --method cdql --seed 0 --out runs/cdql_0
python -m qos_cio evaluate --config configs/default.yaml --eval-seeds 30000:30010 --out runs/classical
python -m qos_cio plot --evaluation runs/demo/evaluation --out runs/replotted \
  --training runs/demo/ppo_seed_0 runs/demo/ppo_seed_1 \
  --ppo-checkpoint runs/demo/ppo_seed_0/policy.pt
```

For learned-policy evaluation, provide `--checkpoints checkpoints.json`:

```json
{"ppo":{"0":"ppo_0/policy.pt"},"cdql":{"0":"cdql_0/policy.pt"}}
```

Paths are resolved relative to the manifest. No checkpoint means evaluate A3 and ReBuHa only. Saved checkpoints support inference and inspection; they are **not exact mid-run resumption snapshots**. Optimizer state is retained for future extensions, but replay contents and simulator RNG state are not saved for resumption.

## Reading the results honestly

`evaluation/episodes.csv` contains one row per policy seed and test trace. `summary.csv` contains means and interval half-widths; `paired_differences.csv` contains matched PPO-minus-CDQL effects. Under each traced evaluation directory, `control.csv`, `users.csv`, `trajectories.csv`, `handovers.csv`, and packet delay/jitter JSON files support audit and replotting.

The absence of a handover is represented by an absent/empty handover event file, not fabricated points. No delivered packets yields missing episode latency, not zero delay. A starved control window receives zero latency utility. Packet conservation is checked explicitly. Pending packets are retained in backlog; they are not silently called delivered or lost.

Full-scale results, trained-to-convergence policies, calibrated standard-compliant radio behavior, and exact numerical reproduction of the manuscript remain unverified. The software does not force PPO to win.

## Authorship and release

The manuscript authors retain authority over its claims and release. This is newly prepared implementation material, not historical code represented as theirs. Natural comments and clear documentation describe engineering choices without inventing provenance. `CITATION.md` explains manuscript citation and the unresolved publication/software attribution. No open-source license has been selected on the authors' behalf; choose one before public redistribution.
