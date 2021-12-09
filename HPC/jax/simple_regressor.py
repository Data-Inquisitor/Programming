import jax
import jax.numpy as jnp
from .loss_functions import LossMixin

class LinearRegressor(LossMixin):
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def predict(self, x):
        return self.w * x + self.b


if __name__ in '__main__':
    my_regressor = LinearRegressor(13., 0.)
    xs = jnp.array([42.0])
    ys = jnp.array([500.0])
    print(my_regressor.rms(xs, ys))
    loss_grad = jax.grad(my_regressor.rms, argnums=(0, 1))
    print(my_regressor.rms(xs, ys))
    print(loss_grad(xs, ys))