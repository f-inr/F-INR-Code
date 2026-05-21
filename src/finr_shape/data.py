from pathlib import Path

import jax.numpy as jnp
import numpy as np


def load_sdf_grid(grid_path: str | Path) -> np.ndarray:
    return np.load(grid_path).astype(np.float32)


def load_gradient_grid(gradient_path: str | Path) -> np.ndarray:
    gradients = np.load(gradient_path).astype(np.float32)
    if gradients.ndim != 4:
        raise ValueError(
            f"Expected gradient grid with 4 dimensions, got shape {gradients.shape}."
        )
    if gradients.shape[-1] == 3:
        return gradients
    if gradients.shape[0] == 3:
        return np.moveaxis(gradients, 0, -1)
    raise ValueError(
        "Expected gradient grid shape `(X, Y, Z, 3)` or `(3, X, Y, Z)`, "
        f"got {gradients.shape}."
    )


def truncate_sdf(grid: np.ndarray, truncation: float = 0.1) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float32)
    return np.where(np.abs(grid) > truncation, np.sign(grid) * truncation, grid)


def downsample_sdf_grid(grid: np.ndarray, stride: int = 1) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float32)
    if stride <= 1:
        return grid
    return grid[::stride, ::stride, ::stride]


def downsample_gradient_grid(gradients: np.ndarray, stride: int = 1) -> np.ndarray:
    gradients = np.asarray(gradients, dtype=np.float32)
    if stride <= 1:
        return gradients
    return gradients[::stride, ::stride, ::stride, :]


def shape_axis_coordinates(grid: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    size_x, size_y, size_z = grid.shape
    x_axis = jnp.linspace(-1.0, 1.0, size_x, dtype=jnp.float32)[:, None]
    y_axis = jnp.linspace(-1.0, 1.0, size_y, dtype=jnp.float32)[:, None]
    z_axis = jnp.linspace(-1.0, 1.0, size_z, dtype=jnp.float32)[:, None]
    return x_axis, y_axis, z_axis


def flattened_shape_coordinates(grid: np.ndarray) -> jnp.ndarray:
    size_x, size_y, size_z = grid.shape
    x_axis = np.linspace(-1.0, 1.0, size_x, dtype=np.float32)
    y_axis = np.linspace(-1.0, 1.0, size_y, dtype=np.float32)
    z_axis = np.linspace(-1.0, 1.0, size_z, dtype=np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    stacked = np.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)],
        axis=1,
    )
    return jnp.asarray(stacked)
