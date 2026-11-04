from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
from common.surrogate import population_stats
from common.td3 import TD3Config, TD3State, TrainFn, init_td3, make_td3_train
from common.utils import genetic_soft_update, h_step_bootstrap
from modules.deep_modules import ActorHead, SharedStateEmbedding
from modules.evo_module import CEM, _cem_ask

Metrics = dict[str, float]
_SHADOW_KEY_OFFSET = 40_000
ADAPT_H_GRID: tuple[int, ...] = (50, 100, 200, 400, 1000)


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

    # P(real full-episode fitness) per generation, else surrogate;
    # 1.0 = pure real evaluation (theta=0.6 was EvoRainbow's h-step coin flip).
    theta: float = 1.0
    h_steps: int = 250

    # short-prefix population evaluation (None = full horizon)
    prefix_steps: int | None = None
    # adaptive prefix h in {50, 100, 200, 400, 1000} calibrated every adapt_interval gens
    prefix_adapt: bool = False
    adapt_interval: int = 20
    adapt_ceiling_every: int = 5
    adapt_window: int = 3
    adapt_r_threshold: float = 0.9
    adapt_min_ceiling: float = 0.3
    # critic-mu pre-screening: ask screen_candidates from CEM, roll out top pop_size
    screen_candidates: int | None = None
    surrogate_batch: int = 1024
    # shadow evaluation cadence (in generations) for prefix / screen metrics
    shadow_interval: int = 10
    # erl-log: log per-individual cumulative returns at h in {50,100,200,400,1000} and episode length
    log_prefixes: bool = False

    # deterministic RL-actor + CEM-elite eval, off the env-step budget
    eval_env_name: str | None = None  # None: env_name
    eval_interval: int = 5_000
    eval_episodes: int = 5


@dataclass
class Run:
    # mutable per-run state shared by ERL and SC-ERL
    cfg: ERLConfig
    horizon: int
    action_limit: float
    buffer: Buffer
    cem: CEM
    unravel_head: Callable[[jax.Array], ActorHead]
    head: CriticHead
    td3_state: TD3State
    train_td3: TrainFn
    eval_env: gym.vector.VectorEnv
    key: jax.Array
    # projects discounted critic values onto the undiscounted return scale
    undiscount_scale: float
    env_steps: int = 0
    generation: int = 0
    next_eval: int = 0
    pending_injection: int | None = None
    closers: list[Callable[[], None]] = field(default_factory=list)

    def close(self) -> None:
        for close in self.closers:
            close()


def env_dims(env_name: str, horizon: int | None) -> tuple[int, int, float, int]:
    env = environments.make_env(env_name)
    try:
        obs_space, action_space = env.observation_space, env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"{env_name}: expected a Box observation space")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"{env_name}: expected a Box action space")
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.shape[0])
        low, high = action_space.low, action_space.high
        if not (np.allclose(high, -low) and np.allclose(high, high[0])):
            raise ValueError(
                f"{env_name}: expected a symmetric, uniform action bound "
                f"(all dims [-b, b]), got low={low}, high={high}"
            )
        action_limit = float(high[0])
        if horizon is None:
            if env.spec is None or env.spec.max_episode_steps is None:
                raise ValueError(
                    f"{env_name}: no max_episode_steps on env.spec; "
                    "pass ERLConfig(horizon=...) explicitly"
                )
            horizon = env.spec.max_episode_steps
    finally:
        env.close()
    return obs_dim, action_dim, action_limit, horizon


def build_run(
    cfg: ERLConfig,
    make_critic_head: Callable[[int, int], CriticHead],
) -> Run:
    obs_dim, action_dim, action_limit, horizon = env_dims(
        cfg.env_name, cfg.horizon
    )
    key = jax.random.key(cfg.seed)
    key, cem_key, td3_key = jax.random.split(key, 3)

    template = ActorHead(
        cfg.embedding_dim, action_dim, key=cem_key, action_limit=action_limit
    )
    flat_template, unravel_head = ravel_pytree(template)
    cem = CEM(
        flat_template.shape[0],
        cem_key,
        sigma_init=cfg.sigma_init,
        pop_size=cfg.pop_size,
        damp=cfg.damp,
        damp_limit=cfg.damp_limit,
        elitisim=cfg.elitism,
    )

    head = make_critic_head(obs_dim, action_dim)
    actor_optimizer = optax.adam(cfg.actor_lr)
    critic_optimizer = optax.adam(cfg.critic_lr)
    td3_state = init_td3(
        obs_dim,
        action_dim,
        cfg.embedding_dim,
        key=td3_key,
        head=head,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
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
    eval_env = environments.make_vec_env(
        cfg.eval_env_name or cfg.env_name,
        2 * cfg.eval_episodes,
        async_=cfg.async_env,
        to_jax=True,
    )
    return Run(
        cfg=cfg,
        horizon=horizon,
        action_limit=action_limit,
        buffer=Buffer(cfg.buffer_capacity, obs_dim, action_dim),
        cem=cem,
        unravel_head=unravel_head,
        head=head,
        td3_state=td3_state,
        train_td3=make_td3_train(
            td3_cfg, head, actor_optimizer, critic_optimizer
        ),
        eval_env=eval_env,
        key=key,
        undiscount_scale=horizon
        * (1.0 - cfg.gamma)
        / (1.0 - cfg.gamma**horizon),
        closers=[eval_env.close],
    )


def next_key(run: Run) -> jax.Array:
    run.key, sub = jax.random.split(run.key)
    return sub


def ask_population(run: Run) -> tuple[jax.Array, ActorHead]:
    flat_pop = run.cem.ask(next_key(run))
    if run.pending_injection is not None:
        rl_flat, _ = ravel_pytree(run.td3_state.online.actor)
        flat_pop = flat_pop.at[run.pending_injection].set(rl_flat)
        run.pending_injection = None
    return flat_pop, jax.vmap(run.unravel_head)(flat_pop)


def train_and_merge(run: Run, gen_env_steps: int) -> Metrics:
    # scales with steps actually collected: a fixed count inflates the
    # replay ratio on cheap generations
    num_updates = int(run.cfg.train_ratio * gen_env_steps)
    td3_state, critic_loss = run.train_td3(
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
    return {
        "train/critic_loss": float(critic_loss),
    }


def maybe_evaluate(run: Run) -> Metrics:
    if run.env_steps < run.next_eval:
        return {}
    run.next_eval = run.env_steps + run.cfg.eval_interval
    heads = jax.tree.map(
        lambda a, b: jnp.stack([a, b]) if eqx.is_array(a) else a,
        run.td3_state.online.actor,
        run.unravel_head(run.cem.elite),
    )
    # fixed key: every eval starts from the same initial states
    rl, elite = evaluate_heads(
        run.eval_env,
        jax.random.key(run.cfg.seed + 10_000),
        run.td3_state.online.embedding,
        heads,
        run.horizon,
    )
    return {"perf/eval_rl": float(rl), "perf/eval_elite": float(elite)}


def to_floats(metrics: dict[str, Any]) -> Metrics:
    return {k: float(v) for k, v in jax.device_get(metrics).items()}


def print_generation(run: Run, metrics: Metrics, tag: str) -> None:
    def show(name: str, fmt: str = "8.1f") -> str:
        value = metrics.get(name)
        return "-" if value is None else format(value, fmt)

    print(
        f"gen {run.generation:4d} | steps {run.env_steps:9d} | {tag} | "
        f"real={show('cost/real_frac', '.2f')} | "
        f"elite_ovl={show('select/elite_overlap', '.2f')} | "
        f"eval_rl={show('perf/eval_rl')} "
        f"eval_elite={show('perf/eval_elite')} | "
        f"critic_loss={show('train/critic_loss', '.4f')}"
    )


@eqx.filter_jit
def rl_policy_step(
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    states: jnp.ndarray,
    key: jax.Array,
    action_limit: float,
    exploration_noise: float,
    warmup: bool,
) -> jnp.ndarray:
    raw = jax.vmap(lambda s: actor(embedding(s)))(states)
    rl_key, warm_key = jax.random.split(key)
    if warmup:
        return jax.random.uniform(
            warm_key, raw.shape, minval=-action_limit, maxval=action_limit
        )
    noise = jax.random.normal(rl_key, raw.shape) * exploration_noise
    return jnp.clip(raw + noise, -action_limit, action_limit)


@eqx.filter_jit
def _step_policy(
    embedding: SharedStateEmbedding,
    pop_heads: ActorHead,
    actor: ActorHead,
    states: jnp.ndarray,
    key: jax.Array,
    action_limit: float,
    exploration_noise: float,
    warmup: bool,
) -> jnp.ndarray:
    z = jax.vmap(embedding)(states)
    pop_actions = jax.vmap(lambda head, zi: head(zi))(pop_heads, z[:-1])

    rl_key, warm_key = jax.random.split(key)
    rl_raw = actor(z[-1])
    if warmup:
        rl_action = jax.random.uniform(
            warm_key, rl_raw.shape, minval=-action_limit, maxval=action_limit
        )
    else:
        noise = jax.random.normal(rl_key, rl_raw.shape) * exploration_noise
        rl_action = jnp.clip(rl_raw + noise, -action_limit, action_limit)

    return jnp.concatenate([pop_actions, rl_action[None]], axis=0)


@eqx.filter_jit
def _deterministic_action(
    embedding: SharedStateEmbedding, actor: ActorHead, state: jnp.ndarray
) -> jnp.ndarray:
    return actor(embedding(state))


@eqx.filter_jit
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
    # onto real_fitness's undiscounted scale; rank-preserving for CEM
    return discounted * undiscount_scale


def _rollout(
    run: Run,
    vec_env: gym.vector.VectorEnv,
    pop_heads: ActorHead,
    warmup: bool,
) -> tuple[jax.Array, jax.Array, Metrics]:
    cfg = run.cfg
    embedding = run.td3_state.online.embedding
    actor = run.td3_state.online.actor

    def policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return _step_policy(
            embedding,
            pop_heads,
            actor,
            states,
            act_key,
            run.action_limit,
            cfg.exploration_noise,
            warmup,
        )

    h_reward = np.zeros((cfg.pop_size + 1, cfg.h_steps), np.float32)
    h_done = np.zeros((cfg.pop_size + 1, cfg.h_steps), np.float32)
    h_state_box: list[jnp.ndarray] = []
    running_ret = np.zeros(cfg.pop_size, dtype=np.float32)
    alive_mask = np.ones(cfg.pop_size, dtype=bool)
    ep_len = np.zeros(cfg.pop_size, dtype=np.int32)
    prefix_snaps: dict[int, np.ndarray] = {}
    grid = tuple(h for h in ADAPT_H_GRID if h < run.horizon) + (run.horizon,)
    grid_set = set(grid)

    def capture_h_step(
        step: int,
        reward: jnp.ndarray,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        next_states: jnp.ndarray,
    ) -> None:
        nonlocal running_ret, alive_mask, ep_len
        if cfg.log_prefixes:
            r_pop = np.asarray(reward[:-1], dtype=np.float32)
            done_pop = np.asarray(terminated[:-1] | truncated[:-1], dtype=bool)
            running_ret = running_ret + r_pop * alive_mask
            ep_len = ep_len + alive_mask.astype(np.int32)
            alive_mask = alive_mask & ~done_pop
            h = step + 1
            if h in grid_set:
                prefix_snaps[h] = running_ret.copy()
        if step >= cfg.h_steps:
            return
        h_reward[:, step] = np.asarray(reward)
        h_done[:, step] = np.asarray(terminated)
        if step == cfg.h_steps - 1:
            h_state_box.append(next_states)

    returns, _ = collect_parallel_episode(
        vec_env,
        next_key(run),
        policy,
        run.buffer,
        run.horizon,
        on_step=capture_h_step,
    )
    surrogate = _surrogate_fitness(
        embedding,
        run.td3_state.online.critic1,
        run.td3_state.online.critic2,
        pop_heads,
        h_state_box[0][:-1],
        jnp.asarray(h_reward[:-1]),
        jnp.asarray(h_done[:-1]),
        cfg.h_steps,
        cfg.gamma,
        run.undiscount_scale,
    )
    truth = returns[:-1]
    prefix_metrics: Metrics = {}
    if cfg.log_prefixes:
        prefix_snaps[run.horizon] = np.asarray(truth, dtype=np.float32)
        for h in grid:
            snap = prefix_snaps[h]
            rho = float(spearman(jnp.asarray(snap), truth))
            prefix_metrics[f"prefix_log/rho_{h}"] = (
                0.0 if np.isnan(rho) else rho
            )
            for i in range(cfg.pop_size):
                prefix_metrics[f"prefix_log/ind{i}_h{h}"] = float(snap[i])
        prefix_metrics["prefix_log/ep_len_mean"] = float(ep_len.mean())
        prefix_metrics["prefix_log/ep_len_min"] = float(ep_len.min())
        prefix_metrics["prefix_log/ep_len_max"] = float(ep_len.max())
        for i in range(cfg.pop_size):
            prefix_metrics[f"prefix_log/ind{i}_ep_len"] = float(ep_len[i])
    return truth, surrogate, prefix_metrics


def _ask_screened_population(
    run: Run, candidates: int
) -> tuple[jax.Array, ActorHead, jax.Array, jax.Array]:
    cfg = run.cfg
    cand_flat = _cem_ask(
        next_key(run),
        run.cem.mu,
        run.cem.cov,
        run.cem.scale,
        candidates,
    )
    cand_heads = jax.vmap(run.unravel_head)(cand_flat)
    states = run.buffer.sample(next_key(run), cfg.surrogate_batch)["state"]
    mu, _ = population_stats(
        run.head.stats,
        run.td3_state.online.embedding,
        run.td3_state.online.critic1,
        cand_heads,
        states,
        next_key(run),
    )
    top_idx = jnp.argsort(-mu)[: cfg.pop_size]
    flat_pop = cand_flat[top_idx]
    if run.cem.elitisim:
        flat_pop = flat_pop.at[-1].set(run.cem.elite)
    if run.pending_injection is not None:
        rl_flat, _ = ravel_pytree(run.td3_state.online.actor)
        flat_pop = flat_pop.at[run.pending_injection].set(rl_flat)
        run.pending_injection = None
    return flat_pop, jax.vmap(run.unravel_head)(flat_pop), cand_flat, mu


def _screen_shadow_metrics(
    run: Run,
    vec_env: gym.vector.VectorEnv,
    cand_flat: jax.Array,
    mu: jax.Array,
    shadow_key: jax.Array,
) -> Metrics:
    # metric-only: evaluate all screened + rejected CEM candidates on
    # independent episodes without touching the replay buffer or env_steps
    cfg = run.cfg
    pop_size = cfg.pop_size
    n_blocks = cand_flat.shape[0] // pop_size
    ep_key = jax.random.fold_in(shadow_key, run.generation)
    block_keys = jax.random.split(ep_key, n_blocks)
    embedding = run.td3_state.online.embedding
    chunks = []
    for b in range(n_blocks):
        block_flat = cand_flat[b * pop_size : (b + 1) * pop_size]
        padded_flat = jnp.concatenate([block_flat, block_flat[:1]], axis=0)
        padded_heads = jax.vmap(run.unravel_head)(padded_flat)

        def policy(
            act_key: jax.Array,
            states: jnp.ndarray,
            heads: ActorHead = padded_heads,
        ) -> jnp.ndarray:
            return heads_policy(embedding, heads, states)

        returns, _ = collect_parallel_episode(
            vec_env, block_keys[b], policy, None, run.horizon
        )
        chunks.append(returns[:-1])

    cand_truth = jnp.concatenate(chunks, axis=0)
    top_mu = jnp.argsort(-mu)[:pop_size]
    screened = cand_truth[top_mu]
    blocks = cand_truth.reshape(n_blocks, pop_size)
    parents = run.cem.parents
    top_screened = jnp.sort(screened)[::-1][:parents].mean()
    top_rand = jnp.sort(blocks, axis=1)[:, ::-1][:, :parents].mean()
    return to_floats(
        {
            "screen/mean_gain": screened.mean() - cand_truth.mean(),
            "screen/elite_gain": top_screened - top_rand,
            "screen/overlap_top10": elite_overlap(mu, cand_truth, pop_size),
            "screen/rank_corr": spearman(mu, cand_truth),
        }
    )


def _generation(
    run: Run, vec_env: gym.vector.VectorEnv, shadow_key: jax.Array
) -> Metrics:
    cfg = run.cfg
    # ERL never truncates: every generation is the same fixed cost.
    gen_env_steps = run.horizon * (cfg.pop_size + 1)
    run.env_steps += gen_env_steps
    warmup = len(run.buffer) < cfg.warmup_steps
    cand_flat: jax.Array | None = None
    mu: jax.Array | None = None
    if warmup or cfg.screen_candidates is None:
        flat_pop, pop_heads = ask_population(run)
    else:
        flat_pop, pop_heads, cand_flat, mu = _ask_screened_population(
            run, cfg.screen_candidates
        )

    truth, surrogate, prefix_metrics = _rollout(run, vec_env, pop_heads, warmup)
    use_real = warmup or bool(jax.random.bernoulli(next_key(run), cfg.theta))
    fitness = truth if use_real else surrogate
    run.cem.tell(fitness, flat_pop)

    metrics: Metrics = {
        "cost/real_frac": float(use_real),
        "screen/cem_std": float(jnp.sqrt(run.cem.cov).mean()),
    } | prefix_metrics
    if not warmup:
        metrics |= to_floats(
            selection_metrics(truth, fitness, surrogate, run.cem.parents)
        )
        if (
            cand_flat is not None
            and mu is not None
            and run.generation % cfg.shadow_interval == 0
        ):
            metrics |= _screen_shadow_metrics(
                run, vec_env, cand_flat, mu, shadow_key
            )
        metrics |= train_and_merge(run, gen_env_steps)
        if run.generation % cfg.rl_to_ea_sync_period == 0:
            if cfg.screen_candidates is not None:
                # flat_pop in the next generation is sorted by -mu with
                # cem.elite at -1; replace the weakest non-elite slot
                run.pending_injection = (
                    cfg.pop_size - 2 if cfg.elitism else cfg.pop_size - 1
                )
            else:
                run.pending_injection = int(jnp.argmin(fitness))
    return metrics


def _prefix_generation(
    run: Run,
    rl_env: gym.vector.VectorEnv,
    pop_env: gym.vector.VectorEnv,
    prefix_steps: int,
    shadow_key: jax.Array,
) -> Metrics:
    cfg = run.cfg
    flat_pop, pop_heads = ask_population(run)
    warmup = len(run.buffer) < cfg.warmup_steps
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

    collect_parallel_episode(
        rl_env, next_key(run), rl_policy, run.buffer, run.horizon
    )
    fitness, _ = collect_parallel_episode(
        pop_env, next_key(run), pop_policy, run.buffer, prefix_steps
    )
    gen_env_steps = run.horizon + cfg.pop_size * prefix_steps
    run.env_steps += gen_env_steps
    run.cem.tell(fitness, flat_pop)

    metrics: Metrics = {
        "cost/real_frac": prefix_steps / run.horizon,
        "screen/cem_std": float(jnp.sqrt(run.cem.cov).mean()),
    }
    if not warmup:
        if run.generation % cfg.shadow_interval == 0:
            ep_key = jax.random.fold_in(shadow_key, run.generation)
            shadow_truth, _ = collect_parallel_episode(
                pop_env, ep_key, pop_policy, None, run.horizon
            )
            metrics |= to_floats(
                selection_metrics(
                    shadow_truth, fitness, fitness, run.cem.parents
                )
            )
        metrics |= train_and_merge(run, gen_env_steps)
        if run.generation % cfg.rl_to_ea_sync_period == 0:
            run.pending_injection = int(jnp.argmin(fitness))
    return metrics


@dataclass
class AdaptState:
    grid: tuple[int, ...]
    current_h: int
    pending_h: int | None = None
    last_ceiling: float = 1.0
    calibration_count: int = 0
    cal_corrs: list[np.ndarray] = field(default_factory=list)
    last_r_vec: np.ndarray = field(default_factory=lambda: np.ones(0))
    cal_overhead_steps: int = 0


def _collect_calibration_rollout(
    run: Run,
    pop_env: gym.vector.VectorEnv,
    pop_policy: Callable[[jax.Array, jnp.ndarray], jnp.ndarray],
    grid: tuple[int, ...],
) -> tuple[jax.Array, dict[int, jax.Array]]:
    running = np.zeros(run.cfg.pop_size, dtype=np.float32)
    alive = np.ones(run.cfg.pop_size, dtype=bool)
    prefix_returns: dict[int, jax.Array] = {}
    grid_set = set(grid)

    def capture_prefixes(
        step: int,
        reward: jnp.ndarray,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        next_states: jnp.ndarray,
    ) -> None:
        nonlocal running, alive
        running = running + np.asarray(reward) * alive
        alive = alive & ~np.asarray(terminated | truncated)
        h = step + 1
        if h in grid_set:
            prefix_returns[h] = jnp.asarray(running.copy())

    full_returns, _ = collect_parallel_episode(
        pop_env,
        next_key(run),
        pop_policy,
        run.buffer,
        run.horizon,
        on_step=capture_prefixes,
    )
    prefix_returns[run.horizon] = full_returns
    return full_returns, prefix_returns


def _update_adapt_state(
    cfg: ERLConfig,
    state: AdaptState,
    full_a: jax.Array,
    prefix_returns: dict[int, jax.Array],
    full_b: jax.Array | None,
) -> None:
    if full_b is not None:
        c_ab = float(spearman(full_a, full_b))
        state.last_ceiling = 0.0 if np.isnan(c_ab) else c_ab

    corrs = np.array(
        [
            0.0
            if np.isnan(c := float(spearman(prefix_returns[h], full_a)))
            else c
            for h in state.grid
        ],
        dtype=np.float32,
    )
    state.cal_corrs.append(corrs)
    if len(state.cal_corrs) > cfg.adapt_window:
        state.cal_corrs = state.cal_corrs[-cfg.adapt_window :]
    mean_corrs = np.mean(np.stack(state.cal_corrs, axis=0), axis=0)

    if state.last_ceiling < cfg.adapt_min_ceiling:
        r_vec = mean_corrs / max(abs(state.last_ceiling), 1e-6)
        candidate_h = state.grid[0]
    else:
        r_vec = mean_corrs / state.last_ceiling
        passing = [
            h
            for h, r_val in zip(state.grid, r_vec, strict=True)
            if r_val >= cfg.adapt_r_threshold
        ]
        candidate_h = passing[0] if passing else state.grid[-1]
    state.last_r_vec = r_vec

    if state.calibration_count == 0:
        state.current_h = candidate_h
        state.pending_h = None
    elif candidate_h == state.current_h:
        state.pending_h = None
    elif state.pending_h == candidate_h:
        state.current_h = candidate_h
        state.pending_h = None
    else:
        state.pending_h = candidate_h
    state.calibration_count += 1


def _adapt_prefix_generation(
    run: Run,
    rl_env: gym.vector.VectorEnv,
    pop_env: gym.vector.VectorEnv,
    state: AdaptState,
    shadow_key: jax.Array,
) -> Metrics:
    cfg = run.cfg
    flat_pop, pop_heads = ask_population(run)
    warmup = len(run.buffer) < cfg.warmup_steps
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

    collect_parallel_episode(
        rl_env, next_key(run), rl_policy, run.buffer, run.horizon
    )

    is_calibration = run.generation % cfg.adapt_interval == 0
    if is_calibration:
        need_ceiling = state.calibration_count % cfg.adapt_ceiling_every == 0
        full_a, prefix_returns = _collect_calibration_rollout(
            run, pop_env, pop_policy, state.grid
        )
        full_b: jax.Array | None = None
        if need_ceiling:
            full_b, _ = collect_parallel_episode(
                pop_env, next_key(run), pop_policy, run.buffer, run.horizon
            )
        _update_adapt_state(cfg, state, full_a, prefix_returns, full_b)
        n_cal_eps = 2 if need_ceiling else 1
        pop_steps = cfg.pop_size * run.horizon * n_cal_eps
        state.cal_overhead_steps += max(
            0, pop_steps - cfg.pop_size * state.current_h
        )
        fitness = full_a
        surr_for_metrics = prefix_returns[state.current_h]
    else:
        fitness, _ = collect_parallel_episode(
            pop_env, next_key(run), pop_policy, run.buffer, state.current_h
        )
        pop_steps = cfg.pop_size * state.current_h
        surr_for_metrics = fitness

    gen_env_steps = run.horizon + pop_steps
    run.env_steps += gen_env_steps
    run.cem.tell(fitness, flat_pop)

    h_idx = state.grid.index(state.current_h)
    r_at_h = (
        float(state.last_r_vec[h_idx]) if len(state.last_r_vec) > h_idx else 1.0
    )
    metrics: Metrics = {
        "cost/real_frac": pop_steps / (cfg.pop_size * run.horizon),
        "screen/cem_std": float(jnp.sqrt(run.cem.cov).mean()),
        "adapt/h": float(state.current_h),
        "adapt/r_h": r_at_h,
        "adapt/ceiling": float(state.last_ceiling),
        "adapt/cal_steps": float(state.cal_overhead_steps),
    }
    for h_val, r_val in zip(state.grid, state.last_r_vec, strict=False):
        metrics[f"adapt/r_{h_val}"] = float(r_val)

    if not warmup:
        if run.generation % cfg.shadow_interval == 0:
            ep_key = jax.random.fold_in(shadow_key, run.generation)
            shadow_truth, _ = collect_parallel_episode(
                pop_env, ep_key, pop_policy, None, run.horizon
            )
            metrics |= to_floats(
                selection_metrics(
                    shadow_truth, fitness, surr_for_metrics, run.cem.parents
                )
            )
        metrics |= train_and_merge(run, gen_env_steps)
        if run.generation % cfg.rl_to_ea_sync_period == 0:
            run.pending_injection = int(jnp.argmin(fitness))
    return metrics


def _validate_config(cfg: ERLConfig, horizon: int) -> None:
    modes_set = (
        int(cfg.prefix_steps is not None)
        + int(cfg.prefix_adapt)
        + int(cfg.screen_candidates is not None)
    )
    if modes_set > 1:
        raise ValueError(
            "at most one of prefix_steps, prefix_adapt, or screen_candidates "
            "may be set"
        )
    if cfg.prefix_steps is not None:
        if not 1 <= cfg.prefix_steps <= horizon:
            raise ValueError(
                f"prefix_steps ({cfg.prefix_steps}) must be in [1, {horizon}]"
            )
    elif not cfg.prefix_adapt and cfg.h_steps > horizon:
        raise ValueError(
            f"h_steps ({cfg.h_steps}) must be <= horizon ({horizon})"
        )
    if cfg.prefix_adapt and (
        cfg.adapt_interval < 1
        or cfg.adapt_ceiling_every < 1
        or cfg.adapt_window < 1
    ):
        raise ValueError(
            "adapt_interval, adapt_ceiling_every, and adapt_window must be >= 1"
        )
    if cfg.screen_candidates is not None and (
        cfg.screen_candidates < cfg.pop_size
        or cfg.screen_candidates % cfg.pop_size != 0
    ):
        raise ValueError(
            f"screen_candidates ({cfg.screen_candidates}) must be a positive "
            f"multiple of pop_size ({cfg.pop_size})"
        )
    if cfg.shadow_interval < 1:
        raise ValueError(
            f"shadow_interval ({cfg.shadow_interval}) must be >= 1"
        )


def train(
    cfg: ERLConfig,
    *,
    on_generation: Callable[[Metrics], None] | None = None,
) -> TD3State:
    run = build_run(
        cfg,
        lambda obs_dim, action_dim: make_head(
            "random", obs_dim, action_dim, cfg.critic_hidden_dims
        ),
    )
    try:
        _validate_config(cfg, run.horizon)
    except ValueError:
        run.close()
        raise

    shadow_key = jax.random.key(cfg.seed + _SHADOW_KEY_OFFSET)
    if cfg.prefix_steps is not None or cfg.prefix_adapt:
        rl_env = environments.make_vec_env(
            cfg.env_name, 1, async_=cfg.async_env, to_jax=True
        )
        pop_env = environments.make_vec_env(
            cfg.env_name, cfg.pop_size, async_=cfg.async_env, to_jax=True
        )
        run.closers += [rl_env.close, pop_env.close]

        if cfg.prefix_adapt:
            tag = "prefix-adapt"
            grid = tuple(h for h in ADAPT_H_GRID if h < run.horizon) + (
                run.horizon,
            )
            adapt_state = AdaptState(
                grid=grid,
                current_h=run.horizon,
                last_r_vec=np.ones(len(grid), dtype=np.float32),
            )

            def step_gen() -> Metrics:
                return _adapt_prefix_generation(
                    run, rl_env, pop_env, adapt_state, shadow_key
                )
        else:
            tag = f"prefix-h{cfg.prefix_steps}"

            def step_gen() -> Metrics:
                assert cfg.prefix_steps is not None
                return _prefix_generation(
                    run, rl_env, pop_env, cfg.prefix_steps, shadow_key
                )
    else:
        tag = (
            f"erl-screen{cfg.screen_candidates}"
            if cfg.screen_candidates is not None
            else ("erl-log" if cfg.log_prefixes else "erl")
        )
        vec_env = environments.make_vec_env(
            cfg.env_name, cfg.pop_size + 1, async_=cfg.async_env, to_jax=True
        )
        run.closers.append(vec_env.close)

        def step_gen() -> Metrics:
            return _generation(run, vec_env, shadow_key)

    try:
        while run.env_steps < cfg.step_budget:
            metrics = step_gen()
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
