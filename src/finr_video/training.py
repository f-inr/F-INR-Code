from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import flattened_video_coordinates, video_axis_coordinates
from .metrics import frame_metrics, video_psnr


@dataclass(frozen=True)
class VideoEncodingConfig:
    steps: int = 50_000
    learning_rate: float = 1e-4
    eval_every: int = 1_000
    seed: int = 0
    eval_frame_index: int = 0
    clip_output: bool = True


@dataclass(frozen=True)
class VideoTrainingRecord:
    step: int
    loss: float
    frame_psnr: float
    frame_ssim: float | None
    video_psnr: float


@dataclass(frozen=True)
class VideoEncodingResult:
    params: object
    reconstruction: np.ndarray
    history: tuple[VideoTrainingRecord, ...]
    elapsed_seconds: float


def _video_loss(model, params, time_axis, height_axis, width_axis, target_video):
    prediction = model.apply({"params": params}, time_axis, height_axis, width_axis)
    return jnp.mean(jnp.square(prediction - target_video))


def _baseline_video_loss(model, params, coordinates, target_video):
    prediction = model.apply({"params": params}, coordinates)
    return jnp.mean(jnp.square(prediction - target_video))


def train_video_model(model, video: np.ndarray, config: VideoEncodingConfig) -> VideoEncodingResult:
    video = np.asarray(video, dtype=np.float32)
    time_axis, height_axis, width_axis = video_axis_coordinates(video)
    target = jnp.asarray(video)

    init_key = jax.random.PRNGKey(config.seed)
    variables = model.init(init_key, time_axis, height_axis, width_axis)
    params = variables["params"]

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state):
        loss, grads = jax.value_and_grad(
            lambda params: _video_loss(
                model,
                params,
                time_axis,
                height_axis,
                width_axis,
                target,
            )
        )(
            current_params
        )
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[VideoTrainingRecord] = []
    start = perf_counter()
    loss = jnp.asarray(0.0)

    for step in range(config.steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        if step % config.eval_every == 0 or step == config.steps:
            reconstruction = np.asarray(model.apply({"params": params}, time_axis, height_axis, width_axis))
            if config.clip_output:
                reconstruction = np.clip(reconstruction, 0.0, 1.0)
            psnr_value, ssim_value = frame_metrics(video, reconstruction, config.eval_frame_index)
            history.append(
                VideoTrainingRecord(
                    step=step,
                    loss=float(loss),
                    frame_psnr=psnr_value,
                    frame_ssim=ssim_value,
                    video_psnr=video_psnr(video, reconstruction),
                )
            )

    elapsed_seconds = perf_counter() - start
    reconstruction = np.asarray(model.apply({"params": params}, time_axis, height_axis, width_axis))
    if config.clip_output:
        reconstruction = np.clip(reconstruction, 0.0, 1.0)

    return VideoEncodingResult(
        params=params,
        reconstruction=reconstruction,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )


def train_baseline_video_model(model, video: np.ndarray, config: VideoEncodingConfig) -> VideoEncodingResult:
    video = np.asarray(video, dtype=np.float32)
    channels, frames, height, width = video.shape
    coordinates = flattened_video_coordinates(video)
    target = jnp.asarray(np.transpose(video, (1, 2, 3, 0)).reshape(-1, channels))

    init_key = jax.random.PRNGKey(config.seed)
    variables = model.init(init_key, coordinates)
    params = variables["params"]

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state):
        loss, grads = jax.value_and_grad(
            lambda params: _baseline_video_loss(model, params, coordinates, target)
        )(
            current_params
        )
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[VideoTrainingRecord] = []
    start = perf_counter()
    loss = jnp.asarray(0.0)

    for step in range(config.steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        if step % config.eval_every == 0 or step == config.steps:
            flat_reconstruction = np.asarray(model.apply({"params": params}, coordinates))
            reconstruction = flat_reconstruction.reshape(frames, height, width, channels)
            reconstruction = np.transpose(reconstruction, (3, 0, 1, 2))
            if config.clip_output:
                reconstruction = np.clip(reconstruction, 0.0, 1.0)
            psnr_value, ssim_value = frame_metrics(video, reconstruction, config.eval_frame_index)
            history.append(
                VideoTrainingRecord(
                    step=step,
                    loss=float(loss),
                    frame_psnr=psnr_value,
                    frame_ssim=ssim_value,
                    video_psnr=video_psnr(video, reconstruction),
                )
            )

    elapsed_seconds = perf_counter() - start
    flat_reconstruction = np.asarray(model.apply({"params": params}, coordinates))
    reconstruction = flat_reconstruction.reshape(frames, height, width, channels)
    reconstruction = np.transpose(reconstruction, (3, 0, 1, 2))
    if config.clip_output:
        reconstruction = np.clip(reconstruction, 0.0, 1.0)

    return VideoEncodingResult(
        params=params,
        reconstruction=reconstruction,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )
