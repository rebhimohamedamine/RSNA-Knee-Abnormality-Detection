"""YAML config loading with `_base_` inheritance.

A config file may declare `_base_: base.yaml` (a path relative to the
*including* file's directory). The base is loaded first, then this file's
keys are deep-merged on top of it (this file wins on conflicts). `_base_` is
resolved recursively, so a base can itself have a `_base_`. The `_base_` key
never appears in the resolved result.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict.
    Dict values are merged key-by-key; anything else (including lists) is
    replaced wholesale by the override's value."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must parse to a mapping, got {type(data)}")
    return data


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving `_base_` inheritance, and return a plain
    fully-resolved dict (no `_base_` key remains)."""
    path = Path(path).resolve()
    raw = _load_raw(path)

    base_rel = raw.pop("_base_", None)
    if base_rel is None:
        return raw

    base_path = (path.parent / base_rel).resolve()
    base_cfg = load_config(base_path)  # recursive: the base may itself have a _base_
    return _deep_merge(base_cfg, raw)


def save_config(cfg: dict, path: str | Path) -> None:
    """Write a fully-resolved config to disk next to a checkpoint/run
    directory, so the run stays reproducible even if the source config files
    change later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Convenience accessor: `get(cfg, "model.slice_attention.enabled")`."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
