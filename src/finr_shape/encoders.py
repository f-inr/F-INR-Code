import jax
import jax.numpy as jnp
from flax import linen as nn


def notebook_frequency_bands(num_frequencies: int, dtype=jnp.float32) -> jax.Array:
    start = int(-(num_frequencies - 1) / 2)
    stop = int((num_frequencies + 1) / 2)
    return 2.0 ** jnp.arange(start, stop, dtype=dtype)


def notebook_positional_encoding(x: jax.Array, num_frequencies: int) -> jax.Array:
    if num_frequencies <= 0:
        return x
    freq = notebook_frequency_bands(num_frequencies, dtype=x.dtype)[None, :]
    return jnp.concatenate((jnp.sin(x @ freq), jnp.cos(x @ freq)), axis=1)


class SimpleHashEncoder1D(nn.Module):
    levels: int = 16
    features_per_level: int = 2
    min_resolution: int = 16
    max_resolution: int = 524_288
    table_size: int = 2**14

    @property
    def growth_factor(self) -> jax.Array:
        return jnp.exp(
            (jnp.log(self.max_resolution) - jnp.log(self.min_resolution))
            / (self.levels - 1)
        )

    @nn.compact
    def __call__(self, x: jax.Array, bound: float = 1.0) -> jax.Array:
        normalized = (x + bound) / (2.0 * bound)
        scales = self.min_resolution * (self.growth_factor ** jnp.arange(self.levels)) - 1
        x_scaled = normalized[:, None] * scales[None, :] + 0.5
        indices = jnp.floor(x_scaled).astype(jnp.int32) % self.table_size

        embeddings = self.param(
            "hash_table",
            lambda key, shape: jax.random.uniform(
                key,
                shape,
                minval=-0.001,
                maxval=0.001,
            ),
            (self.table_size, self.features_per_level),
        )
        return embeddings[indices].reshape(x.shape[0], -1)
