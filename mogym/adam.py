"""Adam with bias correction (Kingma & Ba), the inner solver of all three methods."""
import numpy as np


class Adam:
    def __init__(self, d, lr, beta1=.9, beta2=.999, eps=1e-8):
        self.m = np.zeros(d); self.v = np.zeros(d); self.t = 0; self.lr = lr
        self.beta1 = float(beta1); self.beta2 = float(beta2); self.eps = float(eps)

    def step(self, x, g):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g
        return x - self.lr * (self.m / (1 - self.beta1 ** self.t)) / (
            np.sqrt(self.v / (1 - self.beta2 ** self.t)) + self.eps)
