from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image


def load_image(
    image_path: str | Path,
    downsample_stride: int = 1,
    normalize: bool = True,
) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    if downsample_stride > 1:
        image = image[::downsample_stride, ::downsample_stride, :]
    if normalize:
        image = image / 255.0
    return image


def factorized_coordinates(height: int, width: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_axis = jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32)[:, None]
    y_axis = jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32)[:, None]
    return x_axis, y_axis


def flattened_coordinates(height: int, width: int) -> jnp.ndarray:
    x_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    y_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing="ij")
    stacked = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    return jnp.asarray(stacked)
