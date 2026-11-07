# Do We Need the Critic? Truncated Fitness Evaluation in Budget-Constrained ERL

[![JAX](https://img.shields.io/badge/JAX-0.11+-blue.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.13+-9cf.svg)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

JAX + Equinox ERL (TD3 + CEM population on a shared embedding) with every fitness
estimator behind one `eval_mode` switch: full rollouts, truncated prefixes,
critic bootstrap, TD-adaptive horizons, SC-ERL surrogates and a TD3-only control.

Open work: [TODO.md](TODO.md).

---

## Research question

### The bottleneck
- Step budget $10^6$, population $P=10$, horizon $H=1000$ → one generation costs $11{,}000$ steps → **~90 generations**.
- Prefix $h=200$ → $3{,}000$ steps per generation → **~330 generations** in the same budget.

### Thesis
> Under a fixed environment-step budget, the gain of surrogate-assisted ERL comes
> from cheaper fitness evaluation, not from the critic. A plain truncated prefix
> ($h=200$, no critic in selection) matches SC-ERL; critic bootstrap on the
> prefix ($h$-bootstrap) adds nothing detectable.

### Paper axis
1. **Main comparison**: ERL vs SC-ERL vs prefix-h200, TD3 (`rl_only`) as reference.
2. **Where the gain comes from**: `rl_only` (no population) and `sc_erl_always` controls.
3. **Horizon tradeoff**: $h$ → generations → score (`ablation-horizon`: $h \in \{100, 200, 400, 800\}$).
4. **Does the critic help selection**: `hboot_td` vs `prefix_td`; TD-error controller kept as a diagnosed negative result.

Dropped as contributions: fixing the TD controller, `hboot_td` as the flagship method.

### Results so far (`paper-core-v1`, 4 envs × 10 seeds)
- Normalized final score, IQM [95% CI]: SC-ERL 0.85 [0.67, 1.00], prefix-h200 0.84 [0.67, 0.95], prefix-td 0.81 [0.63, 0.99], hboot-td 0.73 [0.53, 0.92], ERL 0.08 [−0.23, 0.37].
- All cheap estimators ≫ ERL; no separation among them.
- `hboot_td` vs `prefix_td`: P(A>B) = 0.46.
- TD controller saturates ($\delta_{\text{ema}}$ grows with return scale) → $h \approx 700$–$800$, 128–148 generations vs 330.
- Report: https://claude.ai/code/artifact/0a041ade-99cb-4345-b742-48dc4ba03d7f

### Prior negative result
- Critic uncertainty ($\sigma$ from MC-dropout / ensembles / evidential) does not track $Q$-error (AUC ≈ 0.5); uncertainty gating ≈ random gate.

---

## System Architecture

```text
               ┌──────────────────────────────────────────────────┐
               │           SharedStateEmbedding  Z(s)             │
               └────────────────────────┬─────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │   TD3 Gradient Agent  │◄── soft merge (ea_tau) ──┤   CEM ES Population   │
     │      (RL Actor)       │── periodic inject (5 gen)─►│ (10x ActorHead)       │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │ (1x VecEnv, full H)                         │ (10x VecEnv)
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │  Replay Buffer (1M)   │                     │ Fitness Evaluation:   │
     └───────────────────────┘                     │ • Full (H=1000)       │
                 ▲                                 │ • Prefix (h=100-800)  │
                 │ updates (fused JAX JIT)         │ • H-Bootstrap         │
     ┌───────────┴───────────┐                     │ • Adaptive TD Horizon │
     │ Twin Critic Q1 / Q2   │────────────────────►│ • Gated Surrogates    │
     └───────────────────────┘                     └───────────────────────┘
```

---

## Fitness Evaluation Strategies (`eval_mode`)

Selected via `algorithm.eval_mode`:

| Mode | Evaluation | Critic role | Real ES steps per gen |
|---|---|---|---|
| `full` | Full rollout ($H=1000$) | none | $P \times 1000$ |
| `prefix` | Truncated rollout, fixed $h$ (`prefix_steps`) | none | $P \times h$ |
| `prefix_random` | Random $h$ from `prefix_grid` per gen | none | ~42% on average |
| `prefix_td` | Prefix with $h(\delta_{\text{ema}})$ | sets $h$ | dynamic |
| `hboot` | Prefix + $\gamma^h \min(Q_1, Q_2)(s_h)$ | terminal bootstrap | $P \times h$ |
| `hboot_td` | `hboot` with $h(\delta_{\text{ema}})$ | bootstrap + sets $h$ | dynamic |
| `sc_erl_random` | Replay-batch critic scoring, random gate ($\omega=0.79$) | surrogate fitness | ~21% |
| `sc_erl_always` | Surrogate fitness for all members after warmup | surrogate fitness | 0 |
| `ensemble` | $K=5$ critic heads, MAD gate + LCB | surrogate + $\sigma$ gate | varies |
| `semarl_td` | SEMARL TD-error surrogate schedule | surrogate gate | varies |
| `rl_only` | No population, no gene flow: TD3 control | — | 0 (RL actor only) |

- The RL actor always rolls out the full $H$ into the replay buffer (all modes).
- `rl_only` trains TD3 with `train_ratio × H` updates per episode; same networks, budget and evaluation as ERL.

### Adaptive TD-error schedule (`resolve_adaptive_h`)
```python
ratio = 1.0 - math.exp(-max(beta * td_error_ema, 0.0))
target_h = h_min + ratio * (h_max - h_min)
h = min(grid, key=lambda g: abs(g - target_h))
```
- Uses raw (unnormalized) TD error, so `ratio → 1` as returns grow: the controller saturates at $h \approx 800$.

---

## Environments

| Suite | Environments | Role |
|---|---|---|
| **MuJoCo v5** | `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`, `Humanoid-v5` | paper benchmark |
| **DMC Dog** | `dog-stand`, `dog-walk`, `dog-trot`, `dog-run`, `dog-fetch` | registered, not in the paper plan |
| **MyoSuite** | `myoElbowPose1D6MRandom-v0`, `myoHandReachRandom-v0`, `myoHandPenTwirlRandom-v0`, `myoHandObjHoldRandom-v0`, `myoLegWalk-v0` | registered, needs `.venv-myosuite` |

---

## Metrics

### Online (wandb, x-axis `env_steps`)
- `perf/eval_rl`: deterministic RL-actor return, mean of `eval.episodes`.
- `generation`, `env_steps`.
- `cost/real_frac`: real-step fraction used by the population.
- `prefix/h`: current prefix depth (adaptive / random modes).
- `p_surr`: surrogate assignment probability (surrogate modes).
- `train/critic_loss`, `train/td_error`, `train/td_error_ema`.
- `screen/cem_std`: CEM population dispersion.

### Analysis (post-training)
- Final score: mean `perf/eval_rl` over the last 10% of the budget.
- Per-env normalization (0 = worst condition mean, 1 = best); IQM with stratified bootstrap CI over (env, seed).
- P(A>B) per pair; Wilcoxon + Holm across envs (needs ≥ 6 envs).
- AUC over $[0, 10^6]$; steps to threshold and speedup vs ERL.

---

## Quickstart

```bash
uv sync
just train erl HalfCheetah-v5                                        # full ERL
just train erl HalfCheetah-v5 algorithm.eval_mode=prefix algorithm.prefix_steps=200
just train erl HalfCheetah-v5 algorithm.eval_mode=rl_only            # TD3 control
just train sac Hopper-v5                                             # SBX baseline
```

Smoke run:
```bash
uv run python -m train total_steps=3000 algorithm.horizon=100 \
  algorithm.pop_size=4 eval.interval=1000 eval.episodes=1 wandb.enabled=false
```

## Cluster (SLURM on WCSS Lem)

`slurm_run_array.sh` picks conditions via `SUITE`; task ID = condition × 10 + seed.

| Suite | Conditions | Array |
|---|---|---|
| `core` | erl, sc_erl-random, prefix-h200, prefix-td, hboot-td | 0-49 |
| `controls` | rl-only, sc_erl-always | 0-19 |
| `ablation-horizon` | prefix-h100, prefix-h400, prefix-h800 | 0-29 |
| `all` | union of the above | 0-99 |

```bash
uv sync                                                    # login node, once
TARGET_ENV=Humanoid-v5 TAG=paper-core-v1 sbatch --array=0-49 slurm_run_array.sh
./scripts/submit_suite.sh controls HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5
```
- `submit_suite.sh` sets the array range per suite; tag defaults to `paper-<suite>-v1` (override with `TAG=`).
- `WANDB_API_KEY` is read from the environment or `wandb login` / netrc.

---

## Repository Structure

```text
├── CLAUDE.md              # Developer rules and project architecture
├── README.md              # This file
├── TODO.md                # Experiment plan and open work
├── slurm_run_array.sh     # SLURM array (suites: core, controls, ablation-horizon, all)
├── scripts/submit_suite.sh# Submit one suite across envs
├── src/
│   ├── train.py           # Hydra entrypoint
│   ├── runner.py          # Training dispatch and wandb logging
│   ├── algos/erl.py       # ERL loop and all eval modes (incl. rl_only)
│   ├── baselines/         # SBX agents and logging callbacks
│   ├── common/            # Buffer, TD3, rollout, critic heads, surrogate, metrics
│   ├── modules/           # Equinox networks and CEM
│   ├── environments/      # MuJoCo, DMC Dog, MyoSuite registries
│   └── conf/              # Hydra configs
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
