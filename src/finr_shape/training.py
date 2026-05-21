from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import flattened_shape_coordinates, shape_axis_coordinates
from .metrics import calculate_iou, relative_l2


@dataclass(frozen=True)
class ShapeEncodingConfig:
    steps: int = 50_000
    learning_rate: float = 1e-4
    eval_every: int = 5_000
    seed: int = 0
    near_surface_threshold: float = 0.01
    near_surface_weight: float = 1000.0
    use_eikonal: bool = True
    eikonal_weight: float = 1.0
    use_grad_supervision: bool = False
    grad_weight: float = 1.0


@dataclass(frozen=True)
class ShapeTrainingRecord:
    step: int
    loss: float
    relative_l2: float
    iou: float


@dataclass(frozen=True)
class ShapeEncodingResult:
    params: object
    reconstruction: np.ndarray
    history: tuple[ShapeTrainingRecord, ...]
    elapsed_seconds: float


def _shape_loss(
    model,
    params,
    x_axis,
    y_axis,
    z_axis,
    target_grid,
    target_gradients,
    config: ShapeEncodingConfig,
):
    prediction = model.apply({"params": params}, x_axis, y_axis, z_axis)
    loss = jnp.mean(jnp.square(prediction - target_grid))
    predicted_gradients = None

    if config.near_surface_weight > 0:
        mask = jnp.abs(target_grid) < config.near_surface_threshold
        mask_float = mask.astype(prediction.dtype)
        squared_error = jnp.square(prediction - target_grid)
        normalizer = jnp.maximum(jnp.sum(mask_float), 1.0)
        near_surface_error = jnp.sum(squared_error * mask_float) / normalizer
        loss = loss + config.near_surface_weight * near_surface_error

    if config.use_eikonal or config.use_grad_supervision:
        tangent_x = jnp.ones_like(x_axis)
        tangent_y = jnp.ones_like(y_axis)
        tangent_z = jnp.ones_like(z_axis)
        grad_x = jax.jvp(
            lambda axis: model.apply({"params": params}, axis, y_axis, z_axis),
            (x_axis,),
            (tangent_x,),
        )[1]
        grad_y = jax.jvp(
            lambda axis: model.apply({"params": params}, x_axis, axis, z_axis),
            (y_axis,),
            (tangent_y,),
        )[1]
        grad_z = jax.jvp(
            lambda axis: model.apply({"params": params}, x_axis, y_axis, axis),
            (z_axis,),
            (tangent_z,),
        )[1]
        predicted_gradients = jnp.stack([grad_x, grad_y, grad_z], axis=-1)

    if config.use_eikonal:
        grad_norm = jnp.sqrt(jnp.square(grad_x) + jnp.square(grad_y) + jnp.square(grad_z) + 1e-12)
        loss = loss + config.eikonal_weight * jnp.mean(jnp.square(grad_norm - 1.0))

    if config.use_grad_supervision and target_gradients is not None:
        inner_product = jnp.sum(predicted_gradients * target_gradients, axis=-1)
        grad_loss = jnp.mean(jnp.square(1.0 - inner_product))
        loss = loss + config.grad_weight * grad_loss

    return loss


def _baseline_shape_loss(model, params, coordinates, target_grid):
    prediction = model.apply({"params": params}, coordinates)
    return jnp.mean(jnp.square(prediction - target_grid))


def train_shape_model(
    model,
    grid: np.ndarray,
    config: ShapeEncodingConfig,
    gradients: np.ndarray | None = None,
) -> ShapeEncodingResult:
    grid = np.asarray(grid, dtype=np.float32)
    x_axis, y_axis, z_axis = shape_axis_coordinates(grid)
    target = jnp.asarray(grid)
    target_gradients = None if gradients is None else jnp.asarray(gradients, dtype=jnp.float32)

    init_key = jax.random.PRNGKey(config.seed)
    variables = model.init(init_key, x_axis, y_axis, z_axis)
    params = variables["params"]

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state):
        loss, grads = jax.value_and_grad(
            lambda params: _shape_loss(
                model,
                params,
                x_axis,
                y_axis,
                z_axis,
                target,
                target_gradients,
                config,
            )
        )(
            current_params
        )
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[ShapeTrainingRecord] = []
    start = perf_counter()
    loss = jnp.asarray(0.0)

    for step in range(config.steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        if step % config.eval_every == 0 or step == config.steps:
            reconstruction = np.asarray(model.apply({"params": params}, x_axis, y_axis, z_axis))
            history.append(
                ShapeTrainingRecord(
                    step=step,
                    loss=float(loss),
                    relative_l2=relative_l2(reconstruction, grid),
                    iou=calculate_iou(reconstruction, grid),
                )
            )

    elapsed_seconds = perf_counter() - start
    reconstruction = np.asarray(model.apply({"params": params}, x_axis, y_axis, z_axis))
    return ShapeEncodingResult(
        params=params,
        reconstruction=reconstruction,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )


def train_baseline_shape_model(
    model,
    grid: np.ndarray,
    config: ShapeEncodingConfig,
) -> ShapeEncodingResult:
    grid = np.asarray(grid, dtype=np.float32)
    coordinates = flattened_shape_coordinates(grid)
    target = jnp.asarray(grid.reshape(-1))

    init_key = jax.random.PRNGKey(config.seed)
    variables = model.init(init_key, coordinates)
    params = variables["params"]

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state):
        loss, grads = jax.value_and_grad(
            lambda params: _baseline_shape_loss(model, params, coordinates, target)
        )(
            current_params
        )
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[ShapeTrainingRecord] = []
    start = perf_counter()
    loss = jnp.asarray(0.0)

    for step in range(config.steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        if step % config.eval_every == 0 or step == config.steps:
            flat_reconstruction = np.asarray(model.apply({"params": params}, coordinates))
            reconstruction = flat_reconstruction.reshape(grid.shape)
            history.append(
                ShapeTrainingRecord(
                    step=step,
                    loss=float(loss),
                    relative_l2=relative_l2(reconstruction, grid),
                    iou=calculate_iou(reconstruction, grid),
                )
            )

    elapsed_seconds = perf_counter() - start
    flat_reconstruction = np.asarray(model.apply({"params": params}, coordinates))
    reconstruction = flat_reconstruction.reshape(grid.shape)
    return ShapeEncodingResult(
        params=params,
        reconstruction=reconstruction,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )
