from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

import environments
from common.critic_heads import CriticHead, make_head
from common.metrics import elite_overlap, selection_metrics, spearman
from common.replay_buffer import Buffer
from common.rollout import (
    collect_parallel_episode,
    evaluate_heads,
    heads_policy,
)
from common.surrogate import Calibrated, calibrate, gate, make_beta_step, population_stats
from common.td3 import TD3Config, TD3State, TrainFn, init_td3, make_td3_train
from common.utils import genetic_soft_update, h_step_bootstrap
from modules.deep_modules import ActorHead, SharedStateEmbedding
from modules.evo_module import CEM

Metrics = dict[str, float]
_SHADOW_KEY_OFFSET = 40_000

EvalMode = Literal[
    "full",
    "prefix",
    "prefix_random",
    "hboot",
    "sc_erl_random",
    "sc_erl_always",
    "ensemble",
    "semarl_td",
]
EVAL_MODES: tuple[str, ...] = get_args(EvalMode)
PREFIX_GRID: tuple[int, ...] = (100, 200, 400, 600, 800)


@dataclass(frozen=True)
class ERLConfig:
    env_name: str = "Swimmer-v5"
    seed: int = 0
    # env-step budget, not generations: generation cost varies by algorithm
    step_budget: int = 1_000_000
    horizon: int | None = None  # None: use the env's max_episode_steps
    async_env: bool = True

    # Population (CEM over ActorHead's flat params, evaluated on Z(s))
    pop_size: int = 10
    embedding_dim: int = 256
    sigma_init: float = 1e-3
    damp: float = 1e-3
    damp_limit: float = 1e-5
    elitism: bool = True

    # TD3 gradient agent
    critic_hidden_dims: tuple[int, int] = (400, 300)
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    exploration_noise: float = 0.1
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 128
    buffer_capacity: int = 1_000_000
    warmup_steps: int = 10_000
    train_ratio: float = 1.0  # gradient steps per env step collected

    # EA <-> RL gene flow, both gated until the buffer is past warmup
    rl_to_ea_sync_period: int = 5
    ea_tau: float = 0.005

    # Evaluation Mode & Baselines
    eval_mode: str = "full"
    prefix_steps: int | None = None
    prefix_grid: tuple[int, ...] = PREFIX_GRID

    # Gated Surrogate Parameters (SC-ERL & SEMARL)
    omega: float = 0.79  # P(surrogate) for sc_erl_random
    k_ensembles: int = 5  # ensemble members for ensemble mode
    surrogate_batch: int = 1024
    epsilon: float = 0.142
    mad_k: float = 2.483
    beta: float = 0.432
    beta_lr: float = 1e-3

    # SEMARL Adaptive TD-error Gate: p_surr(delta) = clip(alpha * exp(-beta * delta), p_min, p_max)
    semarl_alpha: float = 1.0
    semarl_beta: float = 1.0
    semarl_p_min: float = 0.0
    semarl_p_max: float = 0.9
    semarl_ema_decay: float = 0.1

    # Diagnostics & Evaluation
    shadow_interval: int = 10
    eval_env_name: str | None = None  # None: env_name
    eval_interval: int = 5_000
    eval_episodes: int = 5


@dataclass
class Run:
    cfg: ERLConfig
    action_limit: float
    horizon: int
    undiscount_scale: float
    head: CriticHead
    unravel_head: Callable[[jax.Array], ActorHead]
    cem: CEM
    td3_state: TD3State
    train_td3: TrainFn
    buffer: Buffer
    key: jax.Array
    closers: list[Callable[[], None]] = field(default_factory=list)
    generation: int = 0
    env_steps: int = 0
    pending_injection: int | None = None
    # Running EMA of absolute TD-error (for semarl_td mode)
    td_error_ema: float = 0.0
    # Surrogate calibration state (for gated modes)
    log_beta: jax.Array = field(default_factory=lambda: jnp.asarray(0.0))
    beta_opt_state: Any = None
    beta_step: Any = None
    best_real_return: float = -float("inf")
    best_real_flat: jax.Array | None = None
    pending_anchor: bool = False

    def close(self) -> None:
        for closer in self.closers:
            closer()


def env_dims(env_name: str, horizon: int | None) -> tuple[int, int, float, int]:
    env = environments.make_env(env_name)
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_limit = float(env.action_space.high[0])
        resolved_horizon = (
            int(env.spec.max_episode_steps) if horizon is None else horizon
        )
        return obs_dim, action_dim, action_limit, resolved_horizon
    finally:
        env.close()


def build_run(
    cfg: ERLConfig, head_factory: Callable[[int, int], CriticHead]
) -> Run:
    obs_dim, action_dim, action_limit, horizon = env_dims(
        cfg.env_name, cfg.horizon
    )
    undiscount_scale = (
        horizon if cfg.gamma >= 1.0 else (1.0 - cfg.gamma**horizon) / (1.0 - cfg.gamma)
    )
    key = jax.random.key(cfg.seed)
    key, td3_key, cem_key, buffer_key = jax.random.split(key, 4)

    actor_opt = optax.adam(cfg.actor_lr)
    critic_opt = optax.adam(cfg.critic_lr)
    head = head_factory(obs_dim, action_dim)
    td3_state = init_td3(
        obs_dim,
        action_dim,
        cfg.embedding_dim,
        key=td3_key,
        head=head,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
    )
    td3_cfg = TD3Config(
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        action_limit=action_limit,
        policy_freq=cfg.policy_freq,
        batch_size=cfg.batch_size,
    )
    train_td3 = make_td3_train(td3_cfg, head, actor_opt, critic_opt)

    sample_head = ActorHead(cfg.embedding_dim, action_dim, key=jax.random.key(0))
    flat_sample, unravel_head = ravel_pytree(sample_head)
    cem = CEM(
        key=cem_key,
        num_params=flat_sample.shape[0],
        pop_size=cfg.pop_size,
        sigma_init=cfg.sigma_init,
        damp=cfg.damp,
        damp_limit=cfg.damp_limit,
        elitisim=cfg.elitism,
    )
    buffer = Buffer(
        capacity=cfg.buffer_capacity,
        state_dim=obs_dim,
        action_dim=action_dim,
    )
    beta_opt = optax.adam(cfg.beta_lr)
    return Run(
        cfg=cfg,
        action_limit=action_limit,
        horizon=horizon,
        undiscount_scale=undiscount_scale,
        head=head,
        unravel_head=unravel_head,
        cem=cem,
        td3_state=td3_state,
        train_td3=train_td3,
        buffer=buffer,
        key=key,
        log_beta=jnp.asarray(math.log(max(cfg.beta, 1e-4))),
        beta_opt_state=beta_opt.init(jnp.asarray(math.log(max(cfg.beta, 1e-4)))),
        beta_step=make_beta_step(beta_opt),
    )


def next_key(run: Run) -> jax.Array:
    run.key, k = jax.random.split(run.key)
    return k


def ask_population(run: Run) -> tuple[jax.Array, ActorHead]:
    flat_pop = run.cem.ask(next_key(run))
    if run.pending_injection is not None:
        rl_flat, _ = ravel_pytree(run.td3_state.online.actor)
        flat_pop = flat_pop.at[run.pending_injection].set(rl_flat)
        run.pending_injection = None
    return flat_pop, jax.vmap(run.unravel_head)(flat_pop)


def train_and_merge(run: Run, gen_env_steps: int) -> Metrics:
    num_updates = int(run.cfg.train_ratio * gen_env_steps)
    td3_state, critic_loss, td_err = run.train_td3(
        run.td3_state,
        run.buffer.data,
        jnp.asarray(len(run.buffer)),
        jnp.asarray(num_updates),
        next_key(run),
    )
    champion = run.unravel_head(run.cem.elite)
    actor = genetic_soft_update(
        td3_state.online.actor, champion, run.cfg.ea_tau
    )
    run.td3_state = td3_state._replace(
        online=td3_state.online._replace(actor=actor)
    )
    # Update running EMA of absolute TD-error
    decay = run.cfg.semarl_ema_decay
    run.td_error_ema = (1.0 - decay) * run.td_error_ema + decay * float(td_err)
    return {
        "train/critic_loss": float(critic_loss),
        "train/td_error": float(td_err),
        "train/td_error_ema": float(run.td_error_ema),
    }


def maybe_evaluate(run: Run) -> Metrics:
    eval_env_name = run.cfg.eval_env_name or run.cfg.env_name
    prev = run.env_steps - (
        run.horizon + run.cfg.pop_size * (run.cfg.prefix_steps or run.horizon)
    )
    if prev // run.cfg.eval_interval == run.env_steps // run.cfg.eval_interval:
        return {}
    r = evaluate_actor(
        eval_env_name,
        run.td3_state,
        episodes=run.cfg.eval_episodes,
        seed=run.cfg.seed + 10_000,
    )
    return {"perf/eval_rl": r}


def print_generation(run: Run, metrics: Metrics, tag: str) -> None:
    def show(key: str, fmt: str = ".1f") -> str:
        return f"{metrics[key]:{fmt}}" if key in metrics else "n/a"

    print(
        f"gen {run.generation:4d} | steps {run.env_steps:7d} | "
        f"eval={show('perf/eval_rl')} | "
        f"real_frac={show('cost/real_frac', '.2f')} | "
        f"critic_loss={show('train/critic_loss', '.4f')} | "
        f"td_err={show('train/td_error', '.3f')} (ema={show('train/td_error_ema', '.3f')}) | "
        f"cem_std={show('screen/cem_std', '.4f')} | tag={tag}"
    )


@eqx.filter_jit
def rl_policy_step(
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    states: jnp.ndarray,
    act_key: jax.Array,
    action_limit: float,
    exploration_noise: float,
    warmup: bool,
) -> jnp.ndarray:
    noise_key, uniform_key = jax.random.split(act_key)
    z = embedding(states[0])
    action = actor(z)
    noise = jax.random.normal(noise_key, action.shape) * exploration_noise
    noisy_action = jnp.clip(action + noise, -action_limit, action_limit)
    uniform_action = jax.random.uniform(
        uniform_key, action.shape, minval=-action_limit, maxval=action_limit
    )
    return jnp.where(warmup, uniform_action, noisy_action)[None]


def _deterministic_action(
    embedding: SharedStateEmbedding, actor: ActorHead, state: jnp.ndarray
) -> jnp.ndarray:
    return actor(embedding(state))


def _surrogate_fitness(
    embedding: SharedStateEmbedding,
    critic1: Any,
    critic2: Any,
    pop_heads: ActorHead,
    h_state: jnp.ndarray,
    h_reward: jnp.ndarray,
    h_done: jnp.ndarray,
    h_steps: int,
    gamma: float,
    undiscount_scale: float,
) -> jnp.ndarray:
    z = jax.vmap(embedding)(h_state)
    own_actions = jax.vmap(lambda head, zi: head(zi))(pop_heads, z)
    q1 = jax.vmap(critic1)(h_state, own_actions)[..., 0]
    q2 = jax.vmap(critic2)(h_state, own_actions)[..., 0]
    q_bootstrap = jnp.minimum(q1, q2)
    discounted = h_step_bootstrap(h_steps, gamma, h_reward, h_done, q_bootstrap)
    return discounted * undiscount_scale


# --- Unified Generation Step ---


def _step_generation(
    run: Run,
    rl_env: gym.vector.VectorEnv,
    pop_env: gym.vector.VectorEnv,
    shadow_key: jax.Array,
) -> Metrics:
    cfg = run.cfg
    mode = cfg.eval_mode
    warmup = len(run.buffer) < cfg.warmup_steps
    flat_pop, pop_heads = ask_population(run)
    online = run.td3_state.online

    def rl_policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return rl_policy_step(
            online.embedding,
            online.actor,
            states,
            act_key,
            run.action_limit,
            cfg.exploration_noise,
            warmup,
        )

    def pop_policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return heads_policy(online.embedding, pop_heads, states)

    # 1. RL actor always runs a full horizon rollout into the replay buffer
    collect_parallel_episode(
        rl_env, next_key(run), rl_policy, run.buffer, run.horizon
    )

    metrics: Metrics = {}

    # 2. Warmup: everyone runs full rollout to seed the replay buffer
    if warmup:
        fitness, _ = collect_parallel_episode(
            pop_env, next_key(run), pop_policy, run.buffer, run.horizon
        )
        gen_env_steps = run.horizon * (cfg.pop_size + 1)
        run.env_steps += gen_env_steps
        run.cem.tell(fitness, flat_pop)
        return {
            "cost/real_frac": 1.0,
            "screen/cem_std": float(jnp.sqrt(run.cem.cov).mean()),
        }

    # 3. Dispatched Population Evaluation based on mode
    if mode == "full":
        fitness, _ = collect_parallel_episode(
            pop_env, next_key(run), pop_policy, run.buffer, run.horizon
        )
        pop_steps = cfg.pop_size * run.horizon
        metrics["cost/real_frac"] = 1.0

    elif mode in ("prefix", "prefix_random"):
        if mode == "prefix_random":
            h_idx = int(jax.random.randint(next_key(run), (), 0, len(cfg.prefix_grid)))
            h = cfg.prefix_grid[h_idx]
            metrics["prefix/h"] = float(h)
        else:
            h = cfg.prefix_steps or 200

        fitness, _ = collect_parallel_episode(
            pop_env, next_key(run), pop_policy, run.buffer, h
        )
        pop_steps = cfg.pop_size * h
        metrics["cost/real_frac"] = h / run.horizon

    elif mode == "hboot":
        h = cfg.prefix_steps or 200
        # Capture step rewards and intermediate state at step h
        running_rew = np.zeros((cfg.pop_size, h), dtype=np.float32)
        running_done = np.zeros((cfg.pop_size, h), dtype=np.float32)
        final_state = np.zeros((cfg.pop_size, rl_env.single_observation_space.shape[0]), dtype=np.float32)

        def capture_step(step: int, reward: jnp.ndarray, term: jnp.ndarray, trunc: jnp.ndarray, next_states: jnp.ndarray) -> None:
            running_rew[:, step] = np.asarray(reward)
            running_done[:, step] = np.asarray(term)
            if step == h - 1:
                final_state[:] = np.asarray(next_states)

        collect_parallel_episode(
            pop_env, next_key(run), pop_policy, run.buffer, h, on_step=capture_step
        )
        fitness = _surrogate_fitness(
            online.embedding,
            online.critic1,
            online.critic2,
            pop_heads,
            jnp.asarray(final_state),
            jnp.asarray(running_rew),
            jnp.asarray(running_done),
            h,
            cfg.gamma,
            run.undiscount_scale,
        )
        pop_steps = cfg.pop_size * h
        metrics["cost/real_frac"] = h / run.horizon

    elif mode in ("sc_erl_random", "sc_erl_always", "ensemble", "semarl_td"):
        # Surrogate scoring on replay states
        states = run.buffer.sample(next_key(run), cfg.surrogate_batch)["state"]
        mu, sigma = population_stats(
            run.head.stats,
            online.embedding,
            online.critic1,
            pop_heads,
            states,
            next_key(run),
        )

        # Gating decisions
        if mode == "sc_erl_always":
            real = jnp.zeros(cfg.pop_size, dtype=bool)
            p_surr = 1.0
        elif mode == "sc_erl_random":
            coin = jax.random.uniform(next_key(run), (cfg.pop_size,))
            real = coin > cfg.omega
            p_surr = cfg.omega
        elif mode == "ensemble":
            decision = gate(
                mu, sigma, next_key(run), cfg.omega, cfg.epsilon, cfg.mad_k, True
            )
            real = decision.real
            p_surr = 1.0 - float(jnp.mean(real))
        elif mode == "semarl_td":
            # SEMARL Exponential schedule: p_surr = clip(alpha * exp(-beta * delta_ema), p_min, p_max)
            p_surr = float(
                np.clip(
                    cfg.semarl_alpha * math.exp(-cfg.semarl_beta * run.td_error_ema),
                    cfg.semarl_p_min,
                    cfg.semarl_p_max,
                )
            )
            coin = jax.random.uniform(next_key(run), (cfg.pop_size,))
            real = coin > p_surr

        metrics["p_surr"] = float(p_surr)
        n_real = int(jnp.sum(real))

        # Rollout for real-gated individuals only
        if n_real > 0:
            observed, _ = collect_parallel_episode(
                pop_env,
                next_key(run),
                pop_policy,
                run.buffer,
                run.horizon,
                store_mask=np.asarray(real),
            )
            calibrated = calibrate(
                observed,
                real,
                mu,
                sigma,
                run.log_beta,
                run.undiscount_scale,
            )
            fitness = calibrated.fitness
            pop_steps = n_real * run.horizon
        else:
            # Fully surrogate generation: 0 environment steps spent
            fitness = run.undiscount_scale * (mu - jnp.exp(run.log_beta) * sigma)
            pop_steps = 0

        metrics["cost/real_frac"] = n_real / cfg.pop_size

    else:
        raise ValueError(f"Unknown eval_mode: {mode!r}; choose from {EVAL_MODES}")

    gen_env_steps = run.horizon + pop_steps
    run.env_steps += gen_env_steps
    run.cem.tell(fitness, flat_pop)

    metrics["screen/cem_std"] = float(jnp.sqrt(run.cem.cov).mean())
    metrics |= train_and_merge(run, gen_env_steps)

    if run.generation % cfg.rl_to_ea_sync_period == 0:
        run.pending_injection = int(jnp.argmin(fitness))

    return metrics


def _resolve_eval_mode(cfg: ERLConfig) -> str:
    # Backward compatibility: if prefix_steps is set, infer prefix mode
    if cfg.eval_mode == "full" and cfg.prefix_steps is not None:
        return "prefix"
    return cfg.eval_mode


def train(
    cfg: ERLConfig,
    *,
    on_generation: Callable[[Metrics], None] | None = None,
) -> TD3State:
    mode = _resolve_eval_mode(cfg)
    head_mode = "ensemble" if mode == "ensemble" else "random"
    run = build_run(
        cfg,
        lambda obs_dim, action_dim: make_head(
            head_mode,
            obs_dim,
            action_dim,
            cfg.critic_hidden_dims,
            k_ensembles=cfg.k_ensembles,
        ),
    )

    shadow_key = jax.random.key(cfg.seed + _SHADOW_KEY_OFFSET)
    # Split vec_envs: 1 env for RL actor exploration, pop_size envs for ES population
    rl_env = environments.make_vec_env(
        cfg.env_name, 1, async_=cfg.async_env, to_jax=True
    )
    pop_env = environments.make_vec_env(
        cfg.env_name, cfg.pop_size, async_=cfg.async_env, to_jax=True
    )
    run.closers += [rl_env.close, pop_env.close]

    tag = f"{mode}" if cfg.prefix_steps is None else f"{mode}-h{cfg.prefix_steps}"

    try:
        while run.env_steps < cfg.step_budget:
            metrics = _step_generation(run, rl_env, pop_env, shadow_key)
            metrics |= maybe_evaluate(run)
            metrics |= {
                "generation": float(run.generation),
                "env_steps": float(run.env_steps),
            }
            print_generation(run, metrics, tag)
            if on_generation is not None:
                on_generation(metrics)
            run.generation += 1
    finally:
        run.close()

    return run.td3_state


def evaluate_actor(
    env_name: str,
    td3_state: TD3State,
    *,
    episodes: int = 5,
    seed: int | None = None,
) -> float:
    env = environments.make_env(env_name)
    try:
        total_reward = 0.0
        for i in range(episodes):
            state, _ = env.reset(seed=seed if i == 0 else None)
            done = False
            while not done:
                action = np.asarray(
                    _deterministic_action(
                        td3_state.online.embedding,
                        td3_state.online.actor,
                        jnp.asarray(state),
                    )
                )
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
        return total_reward / episodes
    finally:
        env.close()


if __name__ == "__main__":
    train(ERLConfig())
