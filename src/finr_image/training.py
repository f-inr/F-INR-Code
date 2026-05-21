from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import factorized_coordinates, flattened_coordinates
from .metrics import psnr, ssim_rgb


@dataclass(frozen=True)
class ImageEncodingConfig:
    steps: int = 50_000
    learning_rate: float = 1e-4
    eval_every: int = 1_000
    seed: int = 0
    clip_output: bool = True


@dataclass(frozen=True)
class TrainingRecord:
    step: int
    loss: float
    psnr: float
    ssim: float | None


@dataclass(frozen=True)
class ImageEncodingResult:
    params: object
    reconstruction: np.ndarray
    history: tuple[TrainingRecord, ...]
    elapsed_seconds: float


def _factorized_loss_fn(model, params, x_axis, y_axis, target_image):
    prediction = model.apply({"params": params}, x_axis, y_axis)
    return sum(
        jnp.mean(jnp.square(prediction[:, :, channel] - target_image[:, :, channel]))
        for channel in range(target_image.shape[-1])
    )


def _baseline_loss_fn(model, params, coordinates, target_flat):
    prediction = model.apply({"params": params}, coordinates)
    return jnp.mean(jnp.square(prediction - target_flat))


def train_factorized_image_model(model, image: np.ndarray, config: ImageEncodingConfig) -> ImageEncodingResult:
    image = np.asarray(image, dtype=np.float32)
    x_axis, y_axis = factorized_coordinates(image.shape[0], image.shape[1])
    target = jnp.asarray(image)

    init_key = jax.random.PRNGKey(config.seed)
    variables = model.init(init_key, x_axis, y_axis)
    params = variables["params"]

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state):
        loss, grads = jax.value_and_grad(
            lambda params: _factorized_loss_fn(model, params, x_axis, y_axis, target)
        )(
            current_params
        )
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[TrainingRecord] = []
    start = perf_counter()
    loss = jnp.asarray(0.0)

    for step in range(config.steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        if step % config.eval_every == 0 or step == config.steps:
            reconstruction = np.asarray(model.apply({"params": params}, x_axis, y_axis))
            if config.clip_output:
                reconstruction = np.clip(reconstruction, 0.0, 1.0)
            history.append(
                TrainingRecord(
                    step=step,
                    loss=float(loss),
                    psnr=psnr(image, reconstruction),
                    ssim=ssim_rgb(image, reconstruction),
                )
            )

    elapsed_seconds = perf_counter() - start
    reconstruction = np.asarray(model.apply({"params": params}, x_axis, y_axis))
    if config.clip_output:
        reconstruction = np.clip(reconstruction, 0.0, 1.0)
    return ImageEncodingResult(
        params=params,
        reconstruction=reconstruction,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )


def train_baseline_image_model(model, image: np.ndarray, config: ImageEncodingConfig) -> ImageEncodingResult:
    image = np.asarray(image, dtype=np.float32)
    height, width, channels = image.shape
    coordinates = flattened_coordinates(height, width)
    targets = jnp.asarray(image.reshape(height * width, channels))

    init_key = jax.random.PRNGKey(config.seed)
    variables = model.init(init_key, coordinates)
    params = variables["params"]

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state):
        loss, grads = jax.value_and_grad(
            lambda params: _baseline_loss_fn(model, params, coordinates, targets)
        )(
            current_params
        )
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[TrainingRecord] = []
    start = perf_counter()
    loss = jnp.asarray(0.0)

    for step in range(config.steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        if step % config.eval_every == 0 or step == config.steps:
            flat_reconstruction = np.asarray(model.apply({"params": params}, coordinates))
            reconstruction = flat_reconstruction.reshape(height, width, channels)
            if config.clip_output:
                reconstruction = np.clip(reconstruction, 0.0, 1.0)
            history.append(
                TrainingRecord(
                    step=step,
                    loss=float(loss),
                    psnr=psnr(image, reconstruction),
                    ssim=ssim_rgb(image, reconstruction),
                )
            )

    elapsed_seconds = perf_counter() - start
    flat_reconstruction = np.asarray(model.apply({"params": params}, coordinates))
    reconstruction = flat_reconstruction.reshape(height, width, channels)
    if config.clip_output:
        reconstruction = np.clip(reconstruction, 0.0, 1.0)
    return ImageEncodingResult(
        params=params,
        reconstruction=reconstruction,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )
