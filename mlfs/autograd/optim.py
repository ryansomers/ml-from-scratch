"""First-order optimisers that update ``Tensor.data`` in place from ``Tensor.grad``."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from mlfs.autograd.tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got no parameters")
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional (heavy-ball) momentum.

        v <- mu * v + g
        p <- p - lr * v

    ``weight_decay`` adds lambda * p to the gradient (L2 regularisation).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocity, strict=True):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            if self.momentum:
                v *= self.momentum
                v += g
                g = v
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015).

        m <- b1 m + (1 - b1) g            first-moment (mean) estimate
        v <- b2 v + (1 - b2) g^2          second-moment estimate
        m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)   bias correction
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    m and v start at zero, so early estimates are biased towards zero; dividing
    by (1 - b^t) removes that bias. See docs/derivations.md.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p.data -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
