from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import NavierStokesSuperResolutionData, make_collocation_axes
from .metrics import relative_l2


@dataclass(frozen=True)
class SimulationEncodingConfig:
    steps: int = 20_000
    learning_rate: float = 1e-4
    eval_every: int = 500
    seed: int = 0
    collocation_time_resolution: int = 128
    collocation_space_resolution: int = 128
    offset_count: int = 8
    offset_iter: int = 100
    viscosity: float = 0.01
    divergence_weight: float = 5_000.0
    initial_weight: float = 10_000.0
    observation_weight: float = 10_000.0


@dataclass(frozen=True)
class SimulationTrainingRecord:
    step: int
    loss: float
    relative_l2: float


@dataclass(frozen=True)
class SimulationEncodingResult:
    params: object
    vorticity: np.ndarray
    history: tuple[SimulationTrainingRecord, ...]
    elapsed_seconds: float


def _hvp_fwdfwd(f, primals, tangents):
    inner = lambda primal: jax.jvp(f, (primal,), tangents)[1]
    return jax.jvp(inner, primals, tangents)[1]


def _velocity_to_vorticity(model, params, time_axis, x_axis, y_axis):
    velocity = model.apply({"params": params}, time_axis, x_axis, y_axis)
    vec_x = jnp.ones_like(x_axis)
    vec_y = jnp.ones_like(y_axis)
    v_x = jax.jvp(lambda axis: model.apply({"params": params}, time_axis, axis, y_axis)[1], (x_axis,), (vec_x,))[1]
    u_y = jax.jvp(lambda axis: model.apply({"params": params}, time_axis, x_axis, axis)[0], (y_axis,), (vec_y,))[1]
    return velocity, v_x - u_y


def _residual_loss(model, params, time_axis, x_axis, y_axis, config: SimulationEncodingConfig):
    velocity, vorticity = _velocity_to_vorticity(model, params, time_axis, x_axis, y_axis)
    u_field = velocity[0]
    v_field = velocity[1]

    vec_t = jnp.ones_like(time_axis)
    vec_x = jnp.ones_like(x_axis)
    vec_y = jnp.ones_like(y_axis)

    w_t = jax.jvp(lambda axis: _velocity_to_vorticity(model, params, axis, x_axis, y_axis)[1], (time_axis,), (vec_t,))[1]
    w_x = jax.jvp(lambda axis: _velocity_to_vorticity(model, params, time_axis, axis, y_axis)[1], (x_axis,), (vec_x,))[1]
    w_y = jax.jvp(lambda axis: _velocity_to_vorticity(model, params, time_axis, x_axis, axis)[1], (y_axis,), (vec_y,))[1]
    w_xx = _hvp_fwdfwd(lambda axis: _velocity_to_vorticity(model, params, time_axis, axis, y_axis)[1], (x_axis,), (vec_x,))
    w_yy = _hvp_fwdfwd(lambda axis: _velocity_to_vorticity(model, params, time_axis, x_axis, axis)[1], (y_axis,), (vec_y,))

    u_x = jax.jvp(lambda axis: model.apply({"params": params}, time_axis, axis, y_axis)[0], (x_axis,), (vec_x,))[1]
    v_y = jax.jvp(lambda axis: model.apply({"params": params}, time_axis, x_axis, axis)[1], (y_axis,), (vec_y,))[1]

    pde_residual = w_t + u_field * w_x + v_field * w_y - config.viscosity * (w_xx + w_yy)
    divergence = u_x + v_y
    return jnp.mean(jnp.square(pde_residual)) + config.divergence_weight * jnp.mean(jnp.square(divergence))


def _initial_condition_loss(model, params, data: NavierStokesSuperResolutionData):
    t0 = jnp.asarray(data.t_initial)
    x_hr = jnp.asarray(data.x_hr)
    y_hr = jnp.asarray(data.y_hr)
    velocity, vorticity = _velocity_to_vorticity(model, params, t0, x_hr, y_hr)
    u0_pred = velocity[0, 0]
    v0_pred = velocity[1, 0]
    w0_pred = vorticity[0]
    return (
        jnp.mean(jnp.square(w0_pred - jnp.asarray(data.w0)))
        + jnp.mean(jnp.square(u0_pred - jnp.asarray(data.u0)))
        + jnp.mean(jnp.square(v0_pred - jnp.asarray(data.v0)))
    )


def _observation_loss(model, params, data: NavierStokesSuperResolutionData):
    t_obs = jnp.asarray(data.t_obs)
    x_lr = jnp.asarray(data.x_lr)
    y_lr = jnp.asarray(data.y_lr)
    velocity, vorticity = _velocity_to_vorticity(model, params, t_obs, x_lr, y_lr)
    return (
        jnp.mean(jnp.square(vorticity - jnp.asarray(data.w_obs)))
        + jnp.mean(jnp.square(velocity[0] - jnp.asarray(data.u_obs)))
        + jnp.mean(jnp.square(velocity[1] - jnp.asarray(data.v_obs)))
    )


def _total_loss(model, params, time_axis, x_axis, y_axis, data: NavierStokesSuperResolutionData, config: SimulationEncodingConfig):
    return (
        _residual_loss(model, params, time_axis, x_axis, y_axis, config)
        + config.initial_weight * _initial_condition_loss(model, params, data)
        + config.observation_weight * _observation_loss(model, params, data)
    )


def evaluate_simulation_vorticity(model, params, data: NavierStokesSuperResolutionData) -> np.ndarray:
    _, vorticity = _velocity_to_vorticity(
        model,
        params,
        jnp.asarray(data.t_eval),
        jnp.asarray(data.x_hr),
        jnp.asarray(data.y_hr),
    )
    return np.asarray(vorticity, dtype=np.float32)


def train_simulation_model(model, data: NavierStokesSuperResolutionData, config: SimulationEncodingConfig) -> SimulationEncodingResult:
    collocation_t, collocation_x, collocation_y = make_collocation_axes(
        time_resolution=config.collocation_time_resolution,
        space_resolution=config.collocation_space_resolution,
        offset_count=config.offset_count,
        seed=config.seed,
    )

    init_key = jax.random.PRNGKey(config.seed)
    variables = model.init(init_key, collocation_t[0], collocation_x[0], collocation_y[0])
    params = variables["params"]

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state, time_axis, x_axis, y_axis):
        loss, grads = jax.value_and_grad(
            lambda p: _total_loss(model, p, time_axis, x_axis, y_axis, data, config)
        )(current_params)
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[SimulationTrainingRecord] = []
    start = perf_counter()
    loss = jnp.asarray(0.0)

    for step in range(1, config.steps + 1):
        offset_index = 0
        if config.offset_count > 1:
            offset_index = (step // config.offset_iter) % config.offset_count

        params, opt_state, loss = train_step(
            params,
            opt_state,
            collocation_t[offset_index],
            collocation_x[offset_index],
            collocation_y[offset_index],
        )

        if step % config.eval_every == 0 or step == config.steps:
            vorticity = evaluate_simulation_vorticity(model, params, data)
            history.append(
                SimulationTrainingRecord(
                    step=step,
                    loss=float(loss),
                    relative_l2=relative_l2(vorticity, data.w_eval),
                )
            )

    elapsed_seconds = perf_counter() - start
    vorticity = evaluate_simulation_vorticity(model, params, data)
    return SimulationEncodingResult(
        params=params,
        vorticity=vorticity,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )
