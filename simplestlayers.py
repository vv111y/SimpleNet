import numpy as np


class layer():
    def __init__(self, ffunc, bfunc, node_dim):
        """Should be called by all subclasses."""
        self.forward_fn = ffunc
        self.back_fn = bfunc
        self.input = np.zeros(node_dim)
        self.input_grad = np.zeros(node_dim)

    def add_params(self, *args):
        pass

    def forward(self, x):
        self.input = x
        return self.forward_fn(x)

    def backward(self, error):
        # self.back_params_fn()
        self.input_grad = self.back_fn(error)
        return self.input_grad

    def step(self):
        # apply map to grads, params
        # or send grads & params in order
        pass

    def reset(self):
        self.input.fill(0.)

    def zero_grad(self):
        # self.params.fill(0.)
        self.input_grad.fill(0.)


relu = layer(
    lambda x: np.clip(x, 0, None),
    lambda error: [1 if x > 0 else 0 for x in error],
    10)


sigmoid = layer(
    lambda x: 1 / (1 + np.exp(-1 * x)),
    lambda error: self.forward(error) * (1.0 - self.forward(error)),
    10)

tanh = layer(
    lambda x: np.tanh(x),
    lambda error:  (1 - np.tanh(error)**2),
    10)
