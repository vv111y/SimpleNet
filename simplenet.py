import numpy as np


class FFNN:
    def __init__(self):
        self.net = []
        self.output = None
        self.hold = 0
        self.train = True

    def forward(self, x):
        for layer in self.net:
            if layer.skip:
                self.hold = x
            else:
                x = layer.forward(x) + self.hold
                self.hold = 0
        self.output = x
        return x

    def backward(self, error):
        for layer in reversed(self.net):
            error = layer.backward(error)

    def zero_grad(self):
        for layer in self.net:
            layer.zero_grad()


class layer:
    def __init__(self, node_dim):
        """
        This init should be called via super() with the number
        of nodes as an argument.
        """
        self.input = np.zeros(node_dim)
        self.input_grad = np.zeros(node_dim)
        self.params = False
        self.skip = False

    def forward(self, x):
        self.input = np.copy(x)

    def backward(self):
        pass

    def parameters(self):
        pass

    def zero_grad(self):
        self.input_grad.fill(0.)


class linear(layer):
    def __init__(self, in_dim, node_dim, bias=True):
        super(linear, self).__init__(node_dim)
        self.params = True
        self.bias = bias
        self.input = np.zeros(in_dim)
        self.input_grad = np.zeros(in_dim)
        self.w = np.random.randn(node_dim, in_dim)
        # Alternative random initializations
        # self.w = np.random.rand(node_dim, in_dim)
        # self.w = np.random.uniform(-1/np.sqrt(in_dim),
        #                            1/np.sqrt(in_dim), (node_dim, in_dim))
        self.w_grad = np.zeros((node_dim, in_dim))
        self.b = np.zeros(node_dim)
        self.b_grad = np.zeros(node_dim)

    def forward(self, x):
        self.input = np.copy(x)
        return np.dot(self.w, x) + self.b

    def backward(self, error):
        self.w_grad += np.outer(self.input, error).transpose()
        if self.bias:
            self.b_grad += error
        # self.input_grad += np.dot(np.transpose(self.w), error)
        self.input_grad = np.matmul(error, self.w)
        return self.input_grad

    def parameters(self):
        return zip([self.w, self.b], [self.w_grad, self.b_grad])

    def zero_grad(self):
        self.w_grad.fill(0.)
        self.b_grad.fill(0.)
        self.input_grad.fill(0.)


class conv2d(layer):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, bias=False):
        # call the superclass to initialize attributes common to all layers
        super(conv2d, self).__init__(out_c)

        # set layer attributes 
        self.params = True
        self.bias = bias
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # initialize trainable parameters 
        self.w = np.random.randn(out_c, in_c, kernel_size, kernel_size)
        self.w_grad = np.zeros_like(self.w)
        self.b = np.zeros((out_c, 1, 1))
        self.b_grad = np.zeros(out_c)

    def forward(self, x):
        # save the input for backpropagation later
        self.input = np.copy(x)

        # pad the input 
        x = np.pad(x, [(0, 0), (self.padding, self.padding),
                       (self.padding, self.padding)], 'constant')

        # create the index vectors to loop through
        iY = self.calc_traversal(x[0])
        iX = self.calc_traversal(x[0, 0])

        # instantiate the output tensor
        result = np.zeros((self.out_c, len(iY), len(iX)))

        # perform the convolution 
        for idy in iY:
            for idx in iX:
                result[:, idy, idx] = np.multiply(x[:,
                                                    idy:idy + self.kernel_size,
                                                    idx:idx + self.kernel_size],
                                                  self.w).sum(axis=(1, 2, 3))
        # add the biases and return result
        return self.b + result

    def calc_traversal(self, x):
        # calculate traversal using values of: x.size, kernel.size, stride, padding
        idX = np.arange(
            int((len(x) + 2 * self.padding - self.kernel_size) / self.stride) + 1)
            # int((len(x) + 2 * self.padding - self.kernel_size) / self.stride))

        # change indices based on stride
        idX = idX * self.stride
        return idX

    def backward(self, error):
        iY = self.calc_traversal(error[0])
        iX = self.calc_traversal(error[0][0])
        self.input_grad = np.zeros_like(self.input)

        for idy in iY:
            for idx in iX:
                # (c, y, x)
                self.input_grad[:, idy, idx] = np.multiply(
                    error[:, idy:idy+self.kernel_size, idx:idx+self.kernel_size], self.w)
                self.w_grad[:, idy, idx] = np.multiply(
                    error[:, idy:idy+self.kernel_size, idx:idx+self.kernel_size], self.input)
        # if self.bias:
            # self.b_grad = error
            # self.bias_grad =
        return self.input_grad

    def parameters(self):
        return zip([self.w, self.b], [self.w_grad, self.b_grad])

    def zero_grad(self):
        self.w_grad.fill(0.)
        self.b_grad.fill(0.)
        self.input_grad.fill(0.)


class relu(layer):
    def __init__(self, node_dim):
        super(relu, self).__init__(node_dim)

    def forward(self, x):
        self.input = np.copyy(x)
        return np.clip(x, 0, None)

    def backward(self, error):
        self.input_grad = np.array([1 if x > 0 else 0 for x in self.input])
        return self.input_grad * error


class sigmoid(layer):
    def __init__(self, node_dim):
        super(sigmoid, self).__init__(node_dim)
        self.func = lambda x: 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        self.input = np.copy(x)
        return self.func(x)

    def backward(self, error):
        self.input_grad = self.func(self.input) * (1 - self.func(self.input))
        return self.input_grad * error


class tanh(layer):
    def __init__(self, node_dim):
        # super(tanh, self).__init__(node_dim)
        super().__init__(node_dim)
        self.func = lambda x: (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

    def forward(self, x):
        self.input = np.copy(x)
        return self.func(x)

    def backward(self, error):
        self.input_grad = (1 - self.func(self.input)**2)
        return self.input_grad * error


class softmax(layer):
    def __init__(self, node_dim):
        super(softmax, self).__init__(node_dim)
        # self.func = lambda x: (np.exp(x))/(np.sum((np.exp(x))))
        self.func = lambda x: np.exp(
            x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

    def forward(self, x):
        self.input = x
        return self.func(x)

    def backward(self, error):
        S = self.func(self.input)
        # self.input_grad = self.func(error)*(1 - self.func(error)) 
        # https://github.com/eliben/deep-learning-samples/blob/master/softmax/softmax.py
        self.input_grad = -np.outer(S, S) + np.diag(S.flatten())
        # shape is wrong. nxn * n. need another calc
        return self.input_grad * error


class residual(layer):
    def __init__(self):
        self.skip = True


class dropout(layer):
    def __init__(self, node_dim, rate=0.8):
        super().__init__(node_dim)
        self.rate = rate

    def forward(self, x):
        mask = np.random.binomial(1, self.rate, size=x.shape)
        self.input = np.multiply(x, mask)
        return self.input

    def backward(self, error):
        return np.multiply(error, self.input)


# todo unfinished
class batchnorm(layer):
    def __init__(self, in_dim, affine=True):
        super(batchnorm, self).__init__(in_dim)
        self.affine = affine

        if affine:
            # self.gamma = np.random.randn(in_dim)
            # self.beta = np.random.randn(in_dim)
            self.gamma = np.random.randn(1)
            self.beta = np.random.randn(1)
            self.params = True
        else:
            self.params = False

    def forward(self, x):
        # todo calc
        if self.affine:
            x = self.gamma(x) + self.beta
        return x

    def backward(self, error):
        pass
