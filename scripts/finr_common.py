import json
from pathlib import Path
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_json_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    with path.open() as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def count_params(params: object) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)))


def resolve_output_dir(path_value: str | Path | None, default_subdir: str) -> Path:
    if path_value is None:
        return ROOT / "publishable_runs" / default_subdir
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def maybe_override(current: Any, new_value: Any) -> Any:
    return current if new_value is None else new_value
