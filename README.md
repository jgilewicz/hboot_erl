# HBoot-ERL: Horizon-Bootstrapped & Adaptive Prefix ERL

[![JAX](https://img.shields.io/badge/JAX-0.11+-blue.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.13+-9cf.svg)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HBoot-ERL** is a high-performance, modular library for Evolutionary Reinforcement Learning (ERL) in continuous control, implemented in **JAX** and **Equinox**.

It unifies classical ERL, surrogate-assisted ERL (SC-ERL), truncated prefix evaluations, critic bootstrapping ($h$-bootstrap), and zero-cost adaptive horizon scheduling under a single, unified execution framework.

---

## The Research Problem & Core Hypothesis

### The Bottleneck in Step-Budgeted ERL
In standard Evolutionary Reinforcement Learning (e.g. EvoRainbow, CERL), an off-policy gradient agent (like TD3) shares experience with an evolutionary population (CEM or GA). However, under a strict environment interaction budget (e.g., $10^6$ steps):
* Every generation rolls out all $P$ individuals across the **full episode horizon** ($H=1000$).
* With $P=10$ and $H=1000$, one generation costs $11{,}000$ real environment steps ($1{,}000$ for RL + $10{,}000$ for ES).
* A $1\text{M}$ budget permits **only ~90 generations**, severely starving the evolutionary search of iterations.

### Why Critic Uncertainty Failed
Prior surrogate methods (e.g., SC-ERL) attempted to replace environment rollouts with value function predictions ($Q$-critic) gated by epistemic uncertainty ($\sigma$ from MC-dropout, ensembles, or evidential networks). Extensive empirical testing revealed a **negative result**:
* Epistemic uncertainty does **not** track true value error ($\text{AUC} \approx 0.50$ across environments).
* Uncertainty-gated selection performs no better than a simple random coin flip.

### The Working Hypothesis
> **Truncated prefixes and critic bootstrapping provide a superior, sample-efficient evaluation strategy in budgeted ERL.**
> 1. Evaluating ES individuals on short prefixes ($h \ll H$, e.g., $h=200$) reduces rollout cost by up to $80\%$, enabling **$333$ generations** within $1\text{M}$ steps instead of $90$.
> 2. Bootstrapping the truncated trajectory using the TD3 twin-critic ($h$-bootstrap, $s_h + \gamma^h \min(Q_1, Q_2)$) prevents short-sighted / myopic policy selection.
> 3. Zero-cost adaptive scheduling based on the running TD-error ($\delta_{\text{ema}}$) automatically expands the horizon when the critic is uncalibrated and shortens it when the critic is confident and accurate.

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

All strategies are selected via the `algorithm.eval_mode` Hydra parameter:

| Mode | Evaluation Mechanism | Critic Role | Step Cost per Pop Gen |
|---|---|---|---|
| `full` | Standard full rollout ($H=1000$) | None | $10 \times 1000$ ($100\%$) |
| `prefix` | Truncated rollout to fixed $h \in \{100, 200, 400, 600, 800\}$ | None (pure step sum) | $10 \times h$ ($20\%$ for $h=200$) |
| `prefix_random` | Random horizon $h \in (100, 200, 400, 600, 800)$ per gen | None | $\sim 42\%$ on average |
| `prefix_td` | **Adaptive prefix** via running TD-error $\delta_{\text{ema}}$ | Regulates rollout length $h$ | Dynamic ($100 \le h \le 800$) |
| `hboot` | Truncated rollout ($h=200$) + $h$-step critic bootstrap on $s_h$ | $\gamma^h \min(Q_1, Q_2)$ terminal bootstrap | $10 \times h$ ($20\%$) |
| `hboot_td` | **Adaptive $h$-bootstrap** with dynamic $h(\delta_{\text{ema}})$ | Terminal bootstrap on $s_h$ | Dynamic ($100 \le h \le 800$) |
| `sc_erl_random` | Replay batch scoring; random gate ($\omega=0.79$) | Surrogate scoring + offset calibration | $\sim 21\%$ real rollouts |
| `sc_erl_always` | 100% surrogate scoring after warmup | Full surrogate fitness | $0\%$ real steps |
| `ensemble` | Ensemble critic ($K=5$) with MAD uncertainty gating | Epistemic $\sigma$ gate + LCB scoring | Varies with uncertainty |
| `semarl_td` | SEMARL schedule | TD-error adaptive surrogate gate | Adaptively decreases with error |

### Adaptive TD-Error Schedule (`resolve_adaptive_h`)
```python
ratio = 1.0 - math.exp(-max(beta * td_error_ema, 0.0))
target_h = h_min + ratio * (h_max - h_min)
h = min(grid, key=lambda g: abs(g - target_h))
```
When TD-error is low (critic well-trained), $h \to 100$. When TD-error spikes, $h \to 800$ to maintain ground-truth safety.

---

## Supported Environments (15 Environments)

HBoot-ERL includes built-in wrappers with strict dimension contracts across 3 major benchmark suites:

| Suite | Environments | Obs Dim | Action Dim |
|---|---|---|---|
| **MuJoCo v5** | `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5` | $8 - 27$ | $2 - 8$ |
| **DMC Dog** | `dog-stand`, `dog-walk`, `dog-trot`, `dog-run`, `dog-fetch` | $223 - 232$ | $38$ |
| **MyoSuite** | `myoElbowPose1D6MRandom-v0`, `myoHandReachRandom-v0`, `myoHandPenTwirlRandom-v0`, `myoHandObjHoldRandom-v0`, `myoLegWalk-v0` | $10 - 78$ | $6 - 39$ |

*(DMC environments can also be addressed via the `dm_control/dog-*-v0` alias. MyoSuite requires `.venv-myosuite`)*.

---

## Metrics Tracked

### Online Metrics (Logged to WandB, x-axis: `env_steps`)
* `env_steps`: Total environment interaction steps (RL actor + real population steps).
* `generation`: CEM generation counter.
* `perf/eval_rl`: Deterministic mean episodic return over 5 evaluation episodes.
* `cost/real_frac`: Real step fraction consumed by the ES population ($h/H$ or $N_{\text{real}}/P$).
* `prefix/h`: Current rollout depth $h$ (for adaptive and random modes).
* `p_surr`: Probability of surrogate assignment (for surrogate modes).
* `train/critic_loss`: Mean Bellman MSE loss over gradient updates.
* `train/td_error_ema`: Exponential moving average of TD-error $\delta$.
* `screen/cem_std`: Mean standard deviation of CEM covariance (population dispersion).

### Post-Training Analysis Metrics
* **MA(5)**: Moving average of the last 5 evaluation checkpoints (`perf/eval_rl`) across seeds.
* **Normalized AUC**: Trapezoidal area under the evaluation learning curve over $[0, 10^6]$ steps (sample efficiency).
* **Steps to Threshold & Speedup**: Steps required to achieve $25\%, 50\%, 75\%, 90\%, 100\%$ of baseline return, and relative speedup ratio ($T_{\text{baseline}} / T_{\text{method}}$).
* **Hypothesis Testing**: Pairwise win-rate across 10 random seeds ($\ge 7/10$) and relative percentage gain.
* **Surrogate Diagnostics**: Spearman rank correlation (`select/rank_corr_surr`) and elite overlap (`select/elite_overlap`).

---

## Quickstart

### 1. Installation
The project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management:

```bash
git clone https://github.com/jgilewicz/usc_erl_jax.git
cd usc_erl_jax
uv sync
```

### 2. Running Locally

Train pure ERL:
```bash
just train erl HalfCheetah-v5
```

Train H-Bootstrap ($h=200$):
```bash
just train erl HalfCheetah-v5 algorithm.eval_mode=hboot algorithm.prefix_steps=200
```

Train Adaptive TD Prefix:
```bash
just train erl HalfCheetah-v5 algorithm.eval_mode=prefix_td
```

Train SBX Baselines (SAC, TD3, PPO, CrossQ):
```bash
just train sac Hopper-v5
just train-all Ant-v5
```

### 3. Cluster Benchmarks (SLURM on WCSS)

The script `slurm_run_array.sh` supports modular suites via `SUITE` (default: `core` with 5 conditions x 10 seeds = 50 tasks):

```bash
export WANDB_API_KEY="your-api-key"

# Launch CORE benchmark (5 conditions x 10 seeds = 50 tasks) on Walker2d-v5:
TARGET_ENV=Walker2d-v5 sbatch --array=0-49 slurm_run_array.sh

# Launch all 5 MuJoCo environments on CORE (250 tasks total):
./scripts/submit_mujoco_core.sh

# Launch horizon ablation suite (4 conditions x 10 seeds = 40 tasks):
SUITE=ablation-horizon TARGET_ENV=Walker2d-v5 sbatch --array=0-39 slurm_run_array.sh

# Launch full 14-condition matrix (140 tasks):
SUITE=all TARGET_ENV=Ant-v5 sbatch --array=0-139 slurm_run_array.sh
```

---

## Repository Structure

```text
├── CLAUDE.md              # Compact developer rules and project architecture
├── README.md              # This documentation
├── stan.md                # Condensed system state, thesis, and benchmark matrix
├── pyproject.toml         # Package definition (hboot-erl)
├── slurm_run_array.sh     # Slurm array script (14 conditions x 10 seeds)
├── src/
│   ├── train.py           # Hydra entrypoint
│   ├── runner.py          # Unified training execution and WandB dispatcher
│   ├── algos/
│   │   └── erl.py         # Complete ERL implementation and all evaluation modes
│   ├── baselines/         # Stable-Baselines3/SBX agents and logging callbacks
│   ├── common/            # Buffer, TD3 engine, rollout, critic heads, surrogate, metrics
│   ├── modules/           # Equinox actor/critic networks and CEM evolution
│   ├── environments/      # MuJoCo, DMC Dog, and MyoSuite registries
│   └── conf/              # Hydra configurations (config.yaml, algorithm/*.yaml)
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
