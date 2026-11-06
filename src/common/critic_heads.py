from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from modules.deep_modules import Critic

Mode = Literal["random", "ensemble"]
MODES: tuple[str, ...] = get_args(Mode)

_BOOTSTRAP_P = 0.5
_TARGET_NOISE = 0.02

PointFn = Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array]
LossFn = Callable[[Any, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
StatsFn = Callable[
    [Any, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]


class CriticHead(NamedTuple):
    # critic1 carries the mode's uncertainty (single Critic or ensemble of Critics);
    # critic2 is a plain Critic (twin critic for TD3).
    build: Callable[[jax.Array], Any]
    build_twin: Callable[[jax.Array], Critic]
    point: PointFn
    loss: LossFn
    # per-state (mu, sigma), each (B,)
    stats: StatsFn


def plain_point(
    critic: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> jax.Array:
    keys = jax.random.split(key, states.shape[0])
    return jax.vmap(lambda s, a, k: critic(s, a, key=k))(states, actions, keys)


def plain_loss(
    critic: Critic,
    states: jax.Array,
    actions: jax.Array,
    target: jax.Array,
    key: jax.Array,
) -> jax.Array:
    return jnp.mean((plain_point(critic, states, actions, key) - target) ** 2)


def _no_sigma(
    critic: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    mu = plain_point(critic, states, actions, key)[:, 0]
    return mu, jnp.zeros_like(mu)


def _member_q(
    critics: Critic, states: jax.Array, actions: jax.Array
) -> jax.Array:
    # critics: one Critic whose array leaves carry a leading (K,) axis
    return eqx.filter_vmap(lambda c: jax.vmap(c)(states, actions))(critics)


def _ensemble_point(
    critics: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> jax.Array:
    return _member_q(critics, states, actions).mean(axis=0)


def _ensemble_stats(
    critics: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    q = _member_q(critics, states, actions)[..., 0]
    return q.mean(axis=0), q.std(axis=0, ddof=1)


def _ensemble_loss(
    critics: Critic,
    states: jax.Array,
    actions: jax.Array,
    target: jax.Array,
    key: jax.Array,
) -> jax.Array:
    q = _member_q(critics, states, actions)
    mask_key, noise_key = jax.random.split(key)
    mask = jax.random.bernoulli(mask_key, _BOOTSTRAP_P, q.shape)
    noise = jax.random.normal(noise_key, q.shape)
    noisy_target = target[None] * (1.0 + _TARGET_NOISE * noise)
    loss = optax.huber_loss(q, noisy_target, delta=1.0)
    per_member = jnp.sum(loss * mask, axis=(1, 2)) / (
        jnp.sum(mask, axis=(1, 2)) + 1e-8
    )
    return jnp.sum(per_member)


def make_head(
    mode: str,
    obs_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, int],
    *,
    k_ensembles: int = 5,
) -> CriticHead:
    def plain(key: jax.Array) -> Critic:
        return Critic(
            obs_dim,
            action_dim,
            key=key,
            hidden_dims=hidden_dims,
        )

    match mode:
        case "random":
            return CriticHead(plain, plain, plain_point, plain_loss, _no_sigma)
        case "ensemble":
            def ensemble(key: jax.Array) -> Critic:
                return eqx.filter_vmap(plain)(
                    jax.random.split(key, k_ensembles)
                )

            return CriticHead(
                ensemble,
                plain,
                _ensemble_point,
                _ensemble_loss,
                _ensemble_stats,
            )
        case other:
            raise ValueError(f"unknown surrogate mode {other!r}; use {MODES}")
