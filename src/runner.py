from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from algos.erl import ERLConfig
from algos.erl import evaluate_actor as evaluate_erl_actor
from algos.erl import train as train_erl
from baselines.agents import build_agent
from baselines.envs import build_vec_env
from baselines.wandb_logging import WandbLoggingCallback


def _erl_kwargs(cfg: DictConfig) -> dict[str, Any]:
    algo_cfg = cfg.algorithm
    horizon = int(algo_cfg.horizon)
    pop_size = int(algo_cfg.pop_size)
    kwargs = dict(
        env_name=cfg.env.id,
        seed=cfg.seed,
        step_budget=int(cfg.total_steps),
        horizon=horizon,
        async_env=algo_cfg.async_env,
        pop_size=pop_size,
        embedding_dim=algo_cfg.embedding_dim,
        sigma_init=algo_cfg.sigma_init,
        damp=algo_cfg.damp,
        damp_limit=algo_cfg.damp_limit,
        elitism=algo_cfg.elitism,
        critic_hidden_dims=tuple(algo_cfg.critic_hidden_dims),
        gamma=algo_cfg.gamma,
        tau=algo_cfg.tau,
        policy_noise=algo_cfg.policy_noise,
        noise_clip=algo_cfg.noise_clip,
        policy_freq=algo_cfg.policy_freq,
        exploration_noise=algo_cfg.exploration_noise,
        actor_lr=algo_cfg.actor_lr,
        critic_lr=algo_cfg.critic_lr,
        batch_size=algo_cfg.batch_size,
        buffer_capacity=algo_cfg.buffer_capacity,
        warmup_steps=algo_cfg.warmup_steps,
        train_ratio=algo_cfg.train_ratio,
        rl_to_ea_sync_period=algo_cfg.rl_to_ea_sync_period,
        ea_tau=algo_cfg.ea_tau,
        eval_mode=str(algo_cfg.get("eval_mode", "full")),
        prefix_steps=int(algo_cfg.prefix_steps)
        if algo_cfg.get("prefix_steps") is not None
        else None,
        omega=float(algo_cfg.get("omega", 0.79)),
        k_ensembles=int(algo_cfg.get("k_ensembles", 5)),
        surrogate_batch=int(algo_cfg.get("surrogate_batch", 1024)),
        epsilon=float(algo_cfg.get("epsilon", 0.142)),
        mad_k=float(algo_cfg.get("mad_k", 2.483)),
        beta=float(algo_cfg.get("beta", 0.432)),
        beta_lr=float(algo_cfg.get("beta_lr", 1e-3)),
        semarl_alpha=float(algo_cfg.get("semarl_alpha", 1.0)),
        semarl_beta=float(algo_cfg.get("semarl_beta", 1.0)),
        semarl_p_min=float(algo_cfg.get("semarl_p_min", 0.0)),
        semarl_p_max=float(algo_cfg.get("semarl_p_max", 0.9)),
        semarl_ema_decay=float(algo_cfg.get("semarl_ema_decay", 0.1)),
        shadow_interval=int(algo_cfg.get("shadow_interval", 10)),
        eval_env_name=cfg.eval_env.id,
        eval_interval=int(cfg.eval.interval),
        eval_episodes=int(cfg.eval.episodes),
    )
    # Filter against ERLConfig fields
    valid_fields = set(ERLConfig.__dataclass_fields__.keys())
    return {k: v for k, v in kwargs.items() if k in valid_fields}


def _run_erl(cfg: DictConfig) -> float:
    td3_state = train_erl(
        ERLConfig(**_erl_kwargs(cfg)), on_generation=wandb.log
    )
    return evaluate_erl_actor(
        cfg.eval_env.id,
        td3_state,
        episodes=cfg.eval.episodes,
        seed=cfg.seed + 10_000,
    )


def _run_sc_erl(cfg: DictConfig) -> float:
    # Maps sc_erl mode config into the unified erl runner
    mode = cfg.algorithm.get("mode", "random")
    mapped_eval_mode = f"sc_erl_{mode}" if mode in ("random", "always") else mode
    kwargs = _erl_kwargs(cfg)
    kwargs["eval_mode"] = mapped_eval_mode
    td3_state = train_erl(ERLConfig(**kwargs), on_generation=wandb.log)
    return evaluate_erl_actor(
        cfg.eval_env.id,
        td3_state,
        episodes=cfg.eval.episodes,
        seed=cfg.seed + 10_000,
    )


def _run_sb3(cfg: DictConfig) -> float:
    train_env = build_vec_env(cfg.env.id, cfg.n_envs, cfg.seed)
    eval_env = build_vec_env(cfg.eval_env.id, 1, cfg.seed + 10_000)
    model = build_agent(
        cfg.algorithm, train_env, seed=cfg.seed, device=cfg.device
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=cfg.eval.episodes,
        eval_freq=max(cfg.eval.interval // cfg.n_envs, 1),
        deterministic=True,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=cfg.total_steps,
            callback=CallbackList([WandbLoggingCallback(), eval_cb]),
        )
    finally:
        train_env.close()
        eval_env.close()
    return float(eval_cb.last_mean_reward)


def run_training(cfg: DictConfig) -> float:
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        mode="online" if cfg.wandb.enabled else "disabled",
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
    )
    dispatch = {"erl": _run_erl, "sc_erl": _run_sc_erl}
    if cfg.algorithm.name in dispatch:
        # x-axis is interaction, not iteration: generation cost varies by algorithm
        wandb.define_metric("env_steps")
        wandb.define_metric("*", step_metric="env_steps")
    try:
        runner = dispatch.get(cfg.algorithm.name, _run_sb3)
        eval_reward = runner(cfg)
    except BaseException:
        # a bare `finally: run.finish()` logged crashes as "finished"
        run.finish(exit_code=1)
        raise
    run.finish()

    if cfg.result_file is not None:
        Path(cfg.result_file).write_text(
            json.dumps({"eval_reward": eval_reward})
        )
    return eval_reward
