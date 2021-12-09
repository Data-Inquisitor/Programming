import jax.numpy as jnp

class LossMixin():
    def rms(self, xs: jnp.ndarray, ys: jnp.ndarray):
        return jnp.sqrt(jnp.sum(jnp.square(self.w * xs + self.b - ys)))