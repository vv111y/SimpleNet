import numpy as np

# optim.SGD
# (params,lr=<required parameter>,momentum=0,dampening=0,
# weight_decay=0, nesterov=False)


class SGD():
    def __init__(self, net, N, lr=0.05):
        self.net = net
        self.N = N
        self.lr = lr
        self.eps = np.finfo(float).eps

    def backward(self, error):
        self.net.backward(error / self.N)

    def step(self):
        for l in self.net.net:
            if l.params:
                for p, p_grad in l.parameters():
                    p -= self.lr * p_grad

    def zero_grad(self):
        self.net.zero_grad()

    def MSE(self, y, yhat):
        return 0.5 * np.square(yhat - y)

    def MSE_grad(self, y, yhat):
        return (yhat - y)

    def CE(self, y, yhat):
        yhat = np.clip(yhat, self.eps, 1 - self.eps)
        # y = np.clip(y, self.eps, 1 - self.eps)
        return -1 * (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def CE_grad(self, y, yhat):
        yhat = np.clip(yhat, self.eps, 1 - self.eps)
        # y = np.clip(y, self.eps, 1 - self.eps)
        return -y / yhat + ((1 - y) / (1 - yhat))

    def LogLoss(self, y, yhat):
        yhat = np.clip(yhat, self.eps, 1 - self.eps)
        # y = np.clip(y, self.eps, 1 - self.eps)
        return -y * np.log(yhat)

    def LogLoss_grad(self, y, yhat):
        yhat = np.clip(yhat, self.eps, 1 - self.eps)
        # y = np.clip(y, self.eps, 1 - self.eps)
        return -y / yhat


# shape error. needs adjusting
# return (y * np.log2(yhat)).sum()
# return ((y + self.eps) * np.log2(yhat + self.eps)).sum()
# fastai logloss
# if np.equal(y,[ 1. ]):
#     return -np.log2(yhat)
# else:
#     return -np.log2(1 - yhat)
# # return y * np.log2(yhat)
