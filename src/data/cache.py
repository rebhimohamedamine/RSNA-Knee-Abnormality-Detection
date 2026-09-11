"""On-disk cache for preprocessed per-series tensors, so DICOM decoding
doesn't happen on every epoch.

Layout: `cache/<cache_key>/<StudyUID>/<SeriesUID>.pt` plus a sibling
`<SeriesUID>.metadata.json` holding the same non-tensor fields in a
human-inspectable form. `cache_key` is a hash of exactly the config settings
that affect the cached content (image size, clip percentile, normalization
mode, max slices, sampling strategy) -- changing any of those changes the
key, so a config edit auto-invalidates stale cache instead of silently
serving it. This is a deliberate, narrow deviation from the spec's literal
`cache/<StudyUID>/<SeriesUID>.pt` path, made purely for that auto-invalidation
property; the cache root itself stays configurable via `paths.cache_root`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.data.series import build_series_tensor
from src.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_VERSION = 1

_KEY_FIELDS = ("image_size", "clip_percentile", "normalize", "max_slices", "slice_sampling_strategy")


def cache_key(data_cfg: dict) -> str:
    """Short, stable hash of the subset of `data_cfg` that affects cached
    tensor content."""
    relevant = {field: data_cfg[field] for field in _KEY_FIELDS}
    relevant["clip_percentile"] = list(relevant["clip_percentile"])  # tuples/lists hash the same
    payload = json.dumps(relevant, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def series_cache_path(cache_root: str | Path, key: str, study_uid: str, series_uid: str) -> Path:
    return Path(cache_root) / key / study_uid / f"{series_uid}.pt"


def save_series_cache(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    to_save = dict(payload)
    to_save["cache_version"] = CACHE_VERSION
    torch.save(to_save, path)

    meta = {k: v for k, v in to_save.items() if k != "pixels"}
    meta = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in meta.items()}
    with open(path.with_suffix(".metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_series_cache(path: Path) -> dict | None:
    """Returns the cached payload, or None if missing, unreadable, or from a
    different `CACHE_VERSION` (triggering a rebuild)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        # weights_only=False: this cache holds numpy arrays and plain Python
        # values we wrote ourselves, not untrusted external data.
        data = torch.load(path, weights_only=False)
    except Exception as exc:  # noqa: BLE001 -- any load failure just triggers a rebuild
        logger.warning("Failed to load cache at %s (%s); rebuilding.", path, exc)
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    return data


def get_or_build_series_tensor(
    study_uid: str,
    series_uid: str,
    series_dir: str | Path,
    metadata,
    data_cfg: dict,
    cache_root: str | Path,
) -> dict:
    """Cache-through wrapper around `src.data.series.build_series_tensor`."""
    key = cache_key(data_cfg)
    path = series_cache_path(cache_root, key, study_uid, series_uid)

    cached = load_series_cache(path)
    if cached is not None:
        return cached

    built = build_series_tensor(series_dir, metadata, data_cfg)
    save_series_cache(path, built)
    return built
