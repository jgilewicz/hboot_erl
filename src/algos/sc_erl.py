from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from algos.erl import ERLConfig, Metrics, train as erl_train
from common.td3 import TD3State


@dataclass(frozen=True)
class SCERLConfig(ERLConfig):
    mode: str = "random"


def train(
    cfg: SCERLConfig,
    *,
    on_generation: Callable[[Metrics], None] | None = None,
) -> TD3State:
    eval_mode = (
        f"sc_erl_{cfg.mode}" if cfg.mode in ("random", "always") else cfg.mode
    )
    new_cfg = ERLConfig(
        **{
            k: v
            for k, v in cfg.__dict__.items()
            if k in ERLConfig.__dataclass_fields__ and k != "eval_mode"
        },
        eval_mode=eval_mode,
    )
    return erl_train(new_cfg, on_generation=on_generation)
