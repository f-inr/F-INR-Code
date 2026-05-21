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
    hidden_dims: tuple[int, ...] = (256, 256)
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
                    x = jnp.sin(100.0 * x)
                else:
                    x = nn.Dense(width, kernel_init=siren_hidden_init(30.0))(x)
                    x = jnp.sin(30.0 * x)
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


class CPVideoINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int
    out_channels: int = 3

    @nn.compact
    def __call__(self, time_axis: jax.Array, height_axis: jax.Array, width_axis: jax.Array) -> jax.Array:
        coords = (time_axis, height_axis, width_axis)
        factors = []
        for axis_config, axis_coords in zip(self.axis_configs, coords):
            factor = AxisFeatureExtractor(
                config=axis_config,
                output_dims=(self.out_channels, self.rank),
            )(axis_coords)
            factors.append(factor)
        return jnp.einsum("tor,hor,wor->othw", *factors)


class TuckerVideoINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int
    out_channels: int = 3

    @nn.compact
    def __call__(self, time_axis: jax.Array, height_axis: jax.Array, width_axis: jax.Array) -> jax.Array:
        coords = (time_axis, height_axis, width_axis)
        factors = []
        for axis_config, axis_coords in zip(self.axis_configs, coords):
            factor = AxisFeatureExtractor(
                config=axis_config,
                output_dims=(self.out_channels, self.rank),
            )(axis_coords)
            factors.append(factor)

        core = self.param(
            "core",
            nn.initializers.normal(stddev=0.02),
            (self.out_channels, self.rank, self.rank, self.rank),
        )
        return jnp.einsum("top,hoq,wor,opqr->othw", factors[0], factors[1], factors[2], core)


class TTVideoINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int
    out_channels: int = 3

    @nn.compact
    def __call__(self, time_axis: jax.Array, height_axis: jax.Array, width_axis: jax.Array) -> jax.Array:
        time_factor = AxisFeatureExtractor(
            config=self.axis_configs[0],
            output_dims=(self.out_channels, self.rank, self.rank),
        )(time_axis)
        height_factor = AxisFeatureExtractor(
            config=self.axis_configs[1],
            output_dims=(self.out_channels, self.rank),
        )(height_axis)
        width_factor = AxisFeatureExtractor(
            config=self.axis_configs[2],
            output_dims=(self.out_channels, self.rank),
        )(width_axis)
        return jnp.einsum("topq,hop,woq->othw", time_factor, height_factor, width_factor)


class TRVideoINR(nn.Module):
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig]
    rank: int
    out_channels: int = 3

    @nn.compact
    def __call__(self, time_axis: jax.Array, height_axis: jax.Array, width_axis: jax.Array) -> jax.Array:
        time_factor = AxisFeatureExtractor(
            config=self.axis_configs[0],
            output_dims=(self.out_channels, self.rank, self.rank),
        )(time_axis)
        height_factor = AxisFeatureExtractor(
            config=self.axis_configs[1],
            output_dims=(self.out_channels, self.rank, self.rank),
        )(height_axis)
        width_factor = AxisFeatureExtractor(
            config=self.axis_configs[2],
            output_dims=(self.out_channels, self.rank, self.rank),
        )(width_axis)
        return jnp.einsum("topq,hoqr,worp->othw", time_factor, height_factor, width_factor)


class BaselineVideoINR(nn.Module):
    hidden_dims: tuple[int, ...]
    out_channels: int = 3
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
        time_coords = coordinates[:, :1]
        height_coords = coordinates[:, 1:2]
        width_coords = coordinates[:, 2:3]

        if self.encoding == "positional":
            encoded = jnp.concatenate(
                [
                    notebook_positional_encoding(time_coords, self.pos_enc),
                    notebook_positional_encoding(height_coords, self.pos_enc),
                    notebook_positional_encoding(width_coords, self.pos_enc),
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
                    encoder(time_coords),
                    encoder(height_coords),
                    encoder(width_coords),
                ],
                axis=1,
            )
        elif self.encoding == "raw":
            encoded = coordinates
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")

        if self.backend == "neurbf":
            raise ValueError("NeuRBF is only implemented for factorized video models.")

        return StandardAxisNetwork(
            hidden_dims=self.hidden_dims,
            output_dims=(self.out_channels,),
            backend=self.backend,
        )(encoded)


def build_video_model(
    decomposition: Literal["cp", "tt", "tucker", "tr"],
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig],
    rank: int,
    out_channels: int = 3,
) -> nn.Module:
    if decomposition == "cp":
        return CPVideoINR(axis_configs=axis_configs, rank=rank, out_channels=out_channels)
    if decomposition == "tt":
        return TTVideoINR(axis_configs=axis_configs, rank=rank, out_channels=out_channels)
    if decomposition == "tucker":
        return TuckerVideoINR(axis_configs=axis_configs, rank=rank, out_channels=out_channels)
    if decomposition == "tr":
        return TRVideoINR(axis_configs=axis_configs, rank=rank, out_channels=out_channels)
    raise ValueError(f"Unsupported decomposition: {decomposition}")
