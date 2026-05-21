from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class NavierStokesSuperResolutionData:
    t_eval: np.ndarray
    x_hr: np.ndarray
    y_hr: np.ndarray
    w_eval: np.ndarray
    t_initial: np.ndarray
    w0: np.ndarray
    u0: np.ndarray
    v0: np.ndarray
    t_obs: np.ndarray
    x_lr: np.ndarray
    y_lr: np.ndarray
    w_obs: np.ndarray
    u_obs: np.ndarray
    v_obs: np.ndarray


def _load_snapshot_fields(snapshot_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    snapshot = loadmat(snapshot_path)
    return (
        np.asarray(snapshot["w0"], dtype=np.float32),
        np.asarray(snapshot["u0"], dtype=np.float32),
        np.asarray(snapshot["v0"], dtype=np.float32),
    )


def load_navier_stokes_super_resolution_data(
    data_path: str | Path,
    snapshot_pattern: str,
    snapshot_indices: Sequence[int],
    observation_times: Sequence[float],
    low_res_size: int,
) -> NavierStokesSuperResolutionData:
    if len(snapshot_indices) != len(observation_times):
        raise ValueError("`snapshot_indices` and `observation_times` must have the same length.")

    data_path = Path(data_path)
    root = data_path.parent
    data = loadmat(data_path)

    x_hr = np.asarray(data["x"], dtype=np.float32).reshape(-1, 1)
    y_hr = np.asarray(data["y"], dtype=np.float32).reshape(-1, 1)
    t_high = np.asarray(data["t"], dtype=np.float32).reshape(-1, 1)
    w_high = np.asarray(data["w"], dtype=np.float32)

    w0 = np.asarray(data["w0"], dtype=np.float32)
    u0 = np.asarray(data["u0"], dtype=np.float32)
    v0 = np.asarray(data["v0"], dtype=np.float32)

    full_resolution = int(x_hr.shape[0])
    if full_resolution % low_res_size != 0:
        raise ValueError(
            f"`low_res_size` must divide the full spatial resolution. Got {low_res_size} for {full_resolution}."
        )
    stride = full_resolution // low_res_size

    w_obs = []
    u_obs = []
    v_obs = []
    for index in snapshot_indices:
        snapshot_path = root / snapshot_pattern.format(index=index)
        w_field, u_field, v_field = _load_snapshot_fields(snapshot_path)
        w_obs.append(w_field[::stride, ::stride])
        u_obs.append(u_field[::stride, ::stride])
        v_obs.append(v_field[::stride, ::stride])

    t_eval = np.concatenate([np.zeros((1, 1), dtype=np.float32), t_high], axis=0)
    w_eval = np.concatenate([w0[None, :, :], w_high], axis=0)

    return NavierStokesSuperResolutionData(
        t_eval=t_eval,
        x_hr=x_hr,
        y_hr=y_hr,
        w_eval=w_eval,
        t_initial=np.zeros((1, 1), dtype=np.float32),
        w0=w0,
        u0=u0,
        v0=v0,
        t_obs=np.asarray(observation_times, dtype=np.float32).reshape(-1, 1),
        x_lr=x_hr[::stride],
        y_lr=y_hr[::stride],
        w_obs=np.asarray(w_obs, dtype=np.float32),
        u_obs=np.asarray(u_obs, dtype=np.float32),
        v_obs=np.asarray(v_obs, dtype=np.float32),
    )


def make_collocation_axes(
    time_resolution: int,
    space_resolution: int,
    offset_count: int,
    seed: int,
    time_min: float = 0.0,
    time_max: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 2.0 * np.pi,
    y_min: float = 0.0,
    y_max: float = 2.0 * np.pi,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    time_axis = jnp.linspace(time_min, time_max, time_resolution, endpoint=False, dtype=jnp.float32)[:, None]
    x_axis = jnp.linspace(x_min, x_max, space_resolution, endpoint=False, dtype=jnp.float32)[:, None]
    y_axis = jnp.linspace(y_min, y_max, space_resolution, endpoint=False, dtype=jnp.float32)[:, None]

    if offset_count <= 1:
        return time_axis[None, ...], x_axis[None, ...], y_axis[None, ...]

    key = jax.random.PRNGKey(seed)
    key_t, key_xy = jax.random.split(key)

    dt = (time_max - time_min) / time_resolution
    dxy = (x_max - x_min) / space_resolution

    offsets_t = jax.random.uniform(key_t, (offset_count - 1, 1, 1), minval=0.0, maxval=dt)
    offsets_xy = jax.random.uniform(key_xy, (offset_count - 1, 1, 1), minval=0.0, maxval=dxy)

    time_axes = jnp.concatenate([time_axis[None, ...], time_axis[None, ...] + offsets_t], axis=0)
    x_axes = jnp.concatenate([x_axis[None, ...], x_axis[None, ...] + offsets_xy], axis=0)
    y_axes = jnp.concatenate([y_axis[None, ...], y_axis[None, ...] + offsets_xy], axis=0)
    return time_axes, x_axes, y_axes
