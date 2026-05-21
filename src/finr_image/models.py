from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .encoders import (
    SimpleHashEncoder1D,
    SimpleHashEncoder2D,
    notebook_positional_encoding,
)


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


class RealGaborLayer(nn.Module):
    out_features: int
    omega0: float = 100.0
    sigma0: float = 20.0
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
    hidden_dims: Sequence[int]
    out_dim: int
    backend: str

    @nn.compact
    def __call__(self, coords: jax.Array) -> jax.Array:
        x = coords
        if self.backend == "relu":
            for width in self.hidden_dims:
                x = nn.relu(nn.Dense(width)(x))
            return nn.Dense(self.out_dim)(x)

        if self.backend == "siren":
            for index, width in enumerate(self.hidden_dims):
                if index == 0:
                    x = nn.Dense(width, kernel_init=siren_uniform_init(1.0))(x)
                    x = jnp.sin(100.0 * x)
                else:
                    x = nn.Dense(width, kernel_init=siren_hidden_init(30.0))(x)
                    x = jnp.sin(30.0 * x)
            return nn.Dense(self.out_dim, kernel_init=siren_hidden_init(30.0))(x)

        if self.backend == "siren2":
            for index, width in enumerate(self.hidden_dims):
                omega = 100.0 if index == 0 else 50.0
                x = SineDense(width, omega_0=omega, is_first=index == 0)(x)
            return nn.Dense(self.out_dim)(x)

        if self.backend == "wire":
            for width in self.hidden_dims:
                x = RealGaborLayer(out_features=width)(x)
            return nn.Dense(self.out_dim)(x)

        if self.backend == "finer":
            for width in self.hidden_dims:
                x = nn.Dense(width, kernel_init=finer_init(0.5))(x)
                x = jnp.sin((jnp.abs(x) + 1.0) * x)
            return nn.Dense(self.out_dim, kernel_init=finer_init(0.5))(x)

        raise ValueError(f"Unsupported backend: {self.backend}")


def finer_init(scale: float = 1.0):
    def init(key, shape, dtype=jnp.float32):
        limit = scale / jnp.sqrt(shape[0])
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

    return init


class AxisFeatureExtractor(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    backend: str
    encoding: str = "raw"
    pos_enc: int = 0
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_min_resolution: int = 16
    hash_max_resolution: int = 524_288
    hash_table_size: int = 2**14
    num_rbfs: int = 64

    @nn.compact
    def __call__(self, coords: jax.Array) -> jax.Array:
        encoded = coords
        if self.encoding == "positional":
            encoded = notebook_positional_encoding(coords, self.pos_enc)
        elif self.encoding == "hash":
            encoded = SimpleHashEncoder1D(
                levels=self.hash_levels,
                features_per_level=self.hash_features_per_level,
                min_resolution=self.hash_min_resolution,
                max_resolution=self.hash_max_resolution,
                table_size=self.hash_table_size,
            )(coords)
        elif self.encoding != "raw":
            raise ValueError(f"Unsupported encoding: {self.encoding}")

        if self.backend == "neurbf":
            return NeuRBFAxisNetwork(
                num_rbfs=self.num_rbfs,
                feature_dim=self.out_dim,
            )(encoded)

        return StandardAxisNetwork(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            backend=self.backend,
        )(encoded)


class FactorizedImageINR(nn.Module):
    hidden_dims: Sequence[int]
    rank: int
    out_channels: int = 3
    pos_enc: int = 0
    backend: str = "siren"
    encoding: str = "raw"
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_min_resolution: int = 16
    hash_max_resolution: int = 524_288
    hash_table_size: int = 2**14
    num_rbfs: int = 64

    @nn.compact
    def __call__(self, x_axis: jax.Array, y_axis: jax.Array) -> jax.Array:
        x_features = AxisFeatureExtractor(
            hidden_dims=self.hidden_dims,
            out_dim=self.rank * self.out_channels,
            backend=self.backend,
            encoding=self.encoding,
            pos_enc=self.pos_enc,
            hash_levels=self.hash_levels,
            hash_features_per_level=self.hash_features_per_level,
            hash_min_resolution=self.hash_min_resolution,
            hash_max_resolution=self.hash_max_resolution,
            hash_table_size=self.hash_table_size,
            num_rbfs=self.num_rbfs,
        )(x_axis)
        y_features = AxisFeatureExtractor(
            hidden_dims=self.hidden_dims,
            out_dim=self.rank * self.out_channels,
            backend=self.backend,
            encoding=self.encoding,
            pos_enc=self.pos_enc,
            hash_levels=self.hash_levels,
            hash_features_per_level=self.hash_features_per_level,
            hash_min_resolution=self.hash_min_resolution,
            hash_max_resolution=self.hash_max_resolution,
            hash_table_size=self.hash_table_size,
            num_rbfs=self.num_rbfs,
        )(y_axis)

        # Match the original notebook exactly: each output channel gets its own
        # rank-sized slice, and each slice is merged with a 2D factor contraction.
        x_features = jnp.transpose(x_features, (1, 0))
        y_features = jnp.transpose(y_features, (1, 0))

        channels = []
        for channel in range(self.out_channels):
            start = self.rank * channel
            stop = self.rank * (channel + 1)
            channels.append(
                jnp.einsum(
                    "rh,rw->hw",
                    x_features[start:stop, :],
                    y_features[start:stop, :],
                )
            )

        return jnp.stack(channels, axis=-1)


class BaselineImageINR(nn.Module):
    hidden_dims: Sequence[int]
    out_channels: int = 3
    pos_enc: int = 0
    backend: str = "siren"
    encoding: str = "raw"
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_min_resolution: int = 16
    hash_max_resolution: int = 524_288
    hash_table_size: int = 2**14

    @nn.compact
    def __call__(self, coordinates: jax.Array) -> jax.Array:
        if self.encoding == "hash":
            inputs = SimpleHashEncoder2D(
                levels=self.hash_levels,
                features_per_level=self.hash_features_per_level,
                min_resolution=self.hash_min_resolution,
                max_resolution=self.hash_max_resolution,
                table_size=self.hash_table_size,
            )(coordinates)
        else:
            x_coords = coordinates[:, :1]
            y_coords = coordinates[:, 1:]
            if self.encoding == "positional":
                x_coords = notebook_positional_encoding(x_coords, self.pos_enc)
                y_coords = notebook_positional_encoding(y_coords, self.pos_enc)
            elif self.encoding != "raw":
                raise ValueError(f"Unsupported encoding: {self.encoding}")
            inputs = jnp.concatenate([x_coords, y_coords], axis=1)

        if self.backend == "neurbf":
            raise ValueError("NeuRBF is only implemented for factorized image models.")

        return StandardAxisNetwork(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_channels,
            backend=self.backend,
        )(inputs)
