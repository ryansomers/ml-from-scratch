"""Minimal neural-network building blocks on top of ``Tensor``."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from mlfs.autograd.tensor import Tensor


class Module:
    """Base class: finds parameters by walking attributes (Tensors, Modules, lists)."""

    def parameters(self) -> list[Tensor]:
        return list(self._iter_parameters())

    def _iter_parameters(self) -> Iterator[Tensor]:
        for value in vars(self).values():
            items = value if isinstance(value, list | tuple) else [value]
            for item in items:
                if isinstance(item, Tensor) and item.requires_grad:
                    yield item
                elif isinstance(item, Module):
                    yield from item._iter_parameters()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Linear(Module):
    """Affine map y = x W + b with W of shape (in_features, out_features).

    Weights use He (Kaiming) normal initialisation, std = sqrt(2 / fan_in),
    which keeps activation variance roughly constant through ReLU layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rng: np.random.Generator | None = None,
        dtype: type = np.float64,
    ) -> None:
        rng = rng or np.random.default_rng()
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            (rng.normal(size=(in_features, out_features)) * std).astype(dtype), requires_grad=True
        )
        self.bias = (
            Tensor(np.zeros(out_features, dtype=dtype), requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class MLP(Module):
    """Stack of Linear layers with ReLU between them; the last layer outputs logits.

    ``MLP([784, 128, 10])`` is 784 -> 128 -> ReLU -> 10.
    """

    def __init__(
        self,
        sizes: list[int],
        rng: np.random.Generator | None = None,
        dtype: type = np.float64,
    ) -> None:
        if len(sizes) < 2:
            raise ValueError("need at least input and output sizes")
        rng = rng or np.random.default_rng()
        self.layers = [
            Linear(n_in, n_out, rng=rng, dtype=dtype)
            for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True)
        ]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)
