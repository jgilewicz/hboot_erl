# CLAUDE.md — hboot-erl (usc_erl_jax)

Extends the global CLAUDE.md. Project-specific deltas only.

## Stack

- Python 3.12+, JAX + Equinox + Optax for ERL, [SBX](https://github.com/araffin/sbx) for baselines, Hydra for config, wandb for logging.
- Package manager: `uv`. Quality tools: `ruff`, `ty` (type checking). No pytest suite; verify via smoke runs.

## Layout

```text
src/
  train.py           # Thin Hydra entrypoint (env workers re-import it)
  runner.py          # run_training + dispatch (ERL, SC-ERL, SBX baselines)
  algos/erl.py       # Unified ERL & fitness evaluation (prefix, hboot, gates, adaptive TD)
  baselines/         # SBX agent builders, vec-env wrapping, wandb logging callback
  common/            # Buffer, fused TD3, rollout, critic heads, surrogate gating, metrics, utils
  modules/           # Equinox modules (ActorHead, SharedStateEmbedding) + CEM evolutionary module
  environments/      # Environment registry: mujoco (v5), dmc (dog-*), myosuite
  conf/              # Hydra configs (config.yaml, algorithm/*.yaml)
```

## Conventions

- **Functional JAX style**: Pure functions across `algos/`, `common/`, `modules/`. Classes hold state (e.g. `CEM`), not logic.
- **Unified ERL Runner**: All fitness modes (`full`, `prefix`, `prefix_random`, `prefix_td`, `hboot`, `hboot_td`, `sc_erl_random`, `sc_erl_always`, `ensemble`, `semarl_td`) are handled in [`src/algos/erl.py`](file:///Users/kuba/Home/research/usc_erl_jax/src/algos/erl.py) via `EvalMode`.
- **Gene Flow**:
  - RL $\to$ EA: TD3 actor periodically injected into weakest population member (`rl_to_ea_sync_period = 5`).
  - EA $\to$ RL: Champion softly updates TD3 actor (`ea_tau = 0.005` via `genetic_soft_update`).
- **Rollout Separation**: 1 vectorized env for the RL actor (always full $H$, seeds buffer); `pop_size` vectorized envs for the ES population.
- **Critic Uncertainty**: Closed negative result (AUC $\approx 0.5$, $\sigma$ does not track Q-error). Core focus is prefix truncation, $h$-bootstrap, and TD-error adaptive schedules.
- **Metrics**: WandB x-axis is always `env_steps` (`wandb.define_metric("*", step_metric="env_steps")`). Key metrics: `perf/eval_rl`, `cost/real_frac`, `prefix/h`, `p_surr`, `train/critic_loss`, `train/td_error_ema`, `screen/cem_std`.
- **Verification**: Run `ruff check .`, `ty check`, and a tiny-budget smoke run:
  ```bash
  uv run python -m train total_steps=3000 algorithm.horizon=100 algorithm.pop_size=4 eval.interval=1000 eval.episodes=1 wandb.enabled=false
  ```

## Slurm (WCSS Lem)

- Single benchmark runner: [`slurm_run_array.sh`](file:///Users/kuba/Home/research/usc_erl_jax/slurm_run_array.sh).
- Matrix: 14 conditions $\times$ 10 seeds (0–9) = 140 tasks per target environment.
- Environments: MuJoCo v5 (5 envs), DMC Dog (5 tasks including `dog-fetch`), MyoSuite (5 envs, uses `.venv-myosuite`).
- Run `uv sync` on the login node prior to submissions. Jobs execute `.venv/bin/python` directly on `lem-gpu` (1x NVIDIA H100, 8 CPUs, 16GB RAM).
- Online wandb logging default; syncs to Lustre `$PD_ROOT/wandb_logs` when offline.
