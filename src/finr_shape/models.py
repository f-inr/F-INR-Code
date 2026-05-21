from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from .encoders import SimpleHashEncoder1D, notebook_positional_encoding


BackendName = Literal["relu", "siren", "siren2", "wire", "finer", "neurbf"]
EncodingName = Literal["raw", "positional", "hash"]


@dataclass(frozen=True)
class AxisConfig:
    name: str
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    backend: BackendName = "relu"
    encoding: EncodingName = "raw"
    pos_enc: int = 0
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_min_resolution: int = 16
    hash_max_resolution: int = 524_288
    hash_table_size: int = 2**14
    num_rbfs: int = 64


def siren_uniform_init(scale: float):
    def init(key, shape, dtype=jnp.float32):
        limit = scale / shape[0]
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

    return init


def siren_hidden_init(omega_0: float):
    def init(key, shape, dtype=jnp.float32):
        limit = jnp.sqrt(6.0 / shape[0]) / omega_0
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

    return init


def finer_init(scale: float = 1.0):
    def init(key, shape, dtype=jnp.float32):
        limit = scale / jnp.sqrt(shape[0])
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

    return init


class RealGaborLayer(nn.Module):
    out_features: int
    omega0: float = 10.0
    sigma0: float = 10.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        omega = self.omega0 * nn.Dense(self.out_features, use_bias=self.use_bias)(inputs)
        scale = self.sigma0 * nn.Dense(self.out_features, use_bias=self.use_bias)(inputs)
        return jnp.cos(omega) * jnp.exp(-(scale**2))


class SineDense(nn.Module):
    out_features: int
    omega_0: float
    is_first: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        kernel_init = (
            siren_uniform_init(1.0)
            if self.is_first
            else siren_hidden_init(self.omega_0)
        )
        outputs = nn.Dense(self.out_features, kernel_init=kernel_init)(inputs)
        return jnp.sin(self.omega_0 * outputs)


class NeuRBFAxisNetwork(nn.Module):
    num_rbfs: int
    feature_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        centers = self.param("centers", nn.initializers.uniform(), (self.num_rbfs, 1))
        log_sigma = self.param("log_sigma", nn.initializers.zeros, (self.num_rbfs, 1))
        sigma = jnp.exp(log_sigma) + 1e-6

        freq = self.param(
            "freq",
            nn.initializers.normal(stddev=5.0),
            (1, self.feature_dim),
        )
        bias = self.param("bias", nn.initializers.zeros, (1, self.feature_dim))
        features = self.param(
            "features",
            nn.initializers.normal(stddev=0.1),
            (self.num_rbfs, self.feature_dim),
        )

        x_expanded = x[:, None, :]
        centers_expanded = centers[None, :, :]
        sigma_expanded = sigma[None, :, :]
        sq_dist = ((x_expanded - centers_expanded) ** 2) / (sigma_expanded**2)
        rbf_values = 1.0 / (1.0 + sq_dist.sum(-1))

        composed = jnp.sin(rbf_values[:, :, None] * freq + bias)
        modulated = composed * features[None, :, :]
        aggregated = jnp.sum(modulated, axis=1)

        hidden = nn.Dense(self.feature_dim)(aggregated)
        hidden = jnp.sin(hidden * freq[0]) + hidden
        return nn.Dense(self.feature_dim)(hidden)


class StandardAxisNetwork(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dims: tuple[int, ...]
    backend: BackendName

    @nn.compact
    def __call__(self, coords: jax.Array) -> jax.Array:
        x = coords
        if self.backend == "relu":
            for width in self.hidden_dims:
                x = nn.relu(nn.Dense(width)(x))
        elif self.backend == "siren":
            for index, width in enumerate(self.hidden_dims):
                if index == 0:
                    x = nn.Dense(width, kernel_init=siren_uniform_init(1.0))(x)
                    x = jnp.sin(30.0 * x)
                else:
                    x = nn.Dense(width, kernel_init=siren_hidden_init(30.0))(x)
                    x = jnp.sin(1.0 * x)
        elif self.backend == "siren2":
            for index, width in enumerate(self.hidden_dims):
                omega = 100.0 if index == 0 else 50.0
                x = SineDense(width, omega_0=omega, is_first=index == 0)(x)
        elif self.backend == "wire":
            for width in self.hidden_dims:
                x = RealGaborLayer(out_features=width)(x)
        elif self.backend == "finer":
            for width in self.hidden_dims:
                x = nn.Dense(width, kernel_init=finer_init(0.5))(x)
                x = jnp.sin((jnp.abs(x) + 1.0) * x)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        if len(self.output_dims) == 1:
            return nn.Dense(self.output_dims[0])(x)
        return nn.DenseGeneral(self.output_dims)(x)


class AxisFeatureExtractor(nn.Module):
    config: AxisConfig
    output_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, coords: jax.Array) -> jax.Array:
        encoded = coords
        if self.config.encoding == "positional":
            encoded = notebook_positional_encoding(coords, self.config.pos_enc)
        elif self.config.encoding == "hash":
            encoded = SimpleHashEncoder1D(
                levels=self.config.hash_levels,
                features_per_level=self.config.hash_features_per_level,
                min_resolution=self.config.hash_min_resolution,
                max_resolution=self.config.hash_max_resolution,
                table_size=self.config.hash_table_size,
            )(coords)
        elif self.config.encoding != "raw":
            raise ValueError(f"Unsupported encoding: {self.config.encoding}")

        if self.config.backend == "neurbf":
            if len(self.output_dims) != 1:
                raise ValueError("NeuRBF backend only supports vector outputs.")
            return NeuRBFAxisNetwork(
                num_rbfs=self.config.num_rbfs,
                feature_dim=self.output_dims[0],
            )(encoded)

        return StandardAxisNetwork(
            hidden_dims=self.config.hidden_dims,
            output_dims=self.output_dims,
            backend=self.config.backend,
        )(encoded)


class CPShapeINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int

    @nn.compact
    def __call__(self, x_axis: jax.Array, y_axis: jax.Array, z_axis: jax.Array) -> jax.Array:
        coords = (x_axis, y_axis, z_axis)
        factors = []
        for axis_config, axis_coords in zip(self.axis_configs, coords):
            factor = AxisFeatureExtractor(
                config=axis_config,
                output_dims=(self.rank,),
            )(axis_coords)
            factors.append(factor)
        return jnp.einsum("xr,yr,zr->xyz", *factors)


class TuckerShapeINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int

    @nn.compact
    def __call__(self, x_axis: jax.Array, y_axis: jax.Array, z_axis: jax.Array) -> jax.Array:
        factors = []
        for axis_config, axis_coords in zip(self.axis_configs, (x_axis, y_axis, z_axis)):
            factor = AxisFeatureExtractor(
                config=axis_config,
                output_dims=(self.rank,),
            )(axis_coords)
            factors.append(factor)

        core = self.param(
            "core",
            nn.initializers.normal(stddev=0.02),
            (self.rank, self.rank, self.rank),
        )
        return jnp.einsum("xp,yq,zr,pqr->xyz", factors[0], factors[1], factors[2], core)


class TTShapeINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int

    @nn.compact
    def __call__(self, x_axis: jax.Array, y_axis: jax.Array, z_axis: jax.Array) -> jax.Array:
        x_factor = AxisFeatureExtractor(
            config=self.axis_configs[0],
            output_dims=(self.rank, self.rank),
        )(x_axis)
        y_factor = AxisFeatureExtractor(
            config=self.axis_configs[1],
            output_dims=(self.rank,),
        )(y_axis)
        z_factor = AxisFeatureExtractor(
            config=self.axis_configs[2],
            output_dims=(self.rank,),
        )(z_axis)
        return jnp.einsum("xpr,yp,zr->xyz", x_factor, y_factor, z_factor)


class TRShapeINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int

    @nn.compact
    def __call__(self, x_axis: jax.Array, y_axis: jax.Array, z_axis: jax.Array) -> jax.Array:
        x_factor = AxisFeatureExtractor(
            config=self.axis_configs[0],
            output_dims=(self.rank, self.rank),
        )(x_axis)
        y_factor = AxisFeatureExtractor(
            config=self.axis_configs[1],
            output_dims=(self.rank, self.rank),
        )(y_axis)
        z_factor = AxisFeatureExtractor(
            config=self.axis_configs[2],
            output_dims=(self.rank, self.rank),
        )(z_axis)
        return jnp.einsum("xab,ybc,zca->xyz", x_factor, y_factor, z_factor)


class BaselineShapeINR(nn.Module):
    hidden_dims: tuple[int, ...]
    backend: BackendName = "relu"
    encoding: EncodingName = "raw"
    pos_enc: int = 0
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_min_resolution: int = 16
    hash_max_resolution: int = 524_288
    hash_table_size: int = 2**14

    @nn.compact
    def __call__(self, coordinates: jax.Array) -> jax.Array:
        x_coords = coordinates[:, :1]
        y_coords = coordinates[:, 1:2]
        z_coords = coordinates[:, 2:3]

        if self.encoding == "positional":
            encoded = jnp.concatenate(
                [
                    notebook_positional_encoding(x_coords, self.pos_enc),
                    notebook_positional_encoding(y_coords, self.pos_enc),
                    notebook_positional_encoding(z_coords, self.pos_enc),
                ],
                axis=1,
            )
        elif self.encoding == "hash":
            encoder = SimpleHashEncoder1D(
                levels=self.hash_levels,
                features_per_level=self.hash_features_per_level,
                min_resolution=self.hash_min_resolution,
                max_resolution=self.hash_max_resolution,
                table_size=self.hash_table_size,
            )
            encoded = jnp.concatenate(
                [
                    encoder(x_coords),
                    encoder(y_coords),
                    encoder(z_coords),
                ],
                axis=1,
            )
        elif self.encoding == "raw":
            encoded = coordinates
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")

        if self.backend == "neurbf":
            raise ValueError("NeuRBF is only implemented for factorized shape models.")

        return StandardAxisNetwork(
            hidden_dims=self.hidden_dims,
            output_dims=(1,),
            backend=self.backend,
        )(encoded).squeeze(-1)


def build_shape_model(
    decomposition: Literal["cp", "tt", "tucker", "tr"],
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig],
    rank: int,
) -> nn.Module:
    if decomposition == "cp":
        return CPShapeINR(axis_configs=axis_configs, rank=rank)
    if decomposition == "tt":
        return TTShapeINR(axis_configs=axis_configs, rank=rank)
    if decomposition == "tucker":
        return TuckerShapeINR(axis_configs=axis_configs, rank=rank)
    if decomposition == "tr":
        return TRShapeINR(axis_configs=axis_configs, rank=rank)
    raise ValueError(f"Unsupported decomposition: {decomposition}")
