"""Scalar reverse-mode automatic differentiation.

Every arithmetic operation on ``Value`` objects records its inputs and a
closure that knows the local derivative. ``backward()`` orders the recorded
graph topologically and applies the chain rule from the output back to the
leaves, accumulating d(output)/d(node) into each node's ``grad``.

Gradients accumulate (``+=``) because a node used in several places receives
a contribution from every use (the multivariate chain rule sums over paths).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data: float, _children: Iterable[Value] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _wrap(other: Value | float) -> Value:
        return other if isinstance(other, Value) else Value(other)

    # -- binary operations ------------------------------------------------

    def __add__(self, other: Value | float) -> Value:
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        """Power with a constant (non-Value) exponent: d/dx x^p = p x^(p-1)."""
        if isinstance(exponent, Value):
            raise TypeError("only constant exponents are supported; use exp/log instead")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    # The remaining arithmetic is expressed through the primitives above, so it
    # needs no backward rule of its own.
    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-self._wrap(other))

    def __truediv__(self, other: Value | float) -> Value:
        return self * self._wrap(other) ** -1.0

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __rsub__(self, other: float) -> Value:
        return self._wrap(other) - self

    def __rtruediv__(self, other: float) -> Value:
        return self._wrap(other) / self

    # -- unary functions --------------------------------------------------

    def _unary(self, value: float, local_grad: float, op: str) -> Value:
        out = Value(value, (self,), op)

        def _backward() -> None:
            self.grad += local_grad * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        return self._unary(t, 1.0 - t * t, "tanh")

    def relu(self) -> Value:
        # Subgradient 0 at x = 0.
        return self._unary(max(self.data, 0.0), float(self.data > 0), "relu")

    def exp(self) -> Value:
        e = math.exp(self.data)
        return self._unary(e, e, "exp")

    def log(self) -> Value:
        return self._unary(math.log(self.data), 1.0 / self.data, "log")

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        return self._unary(s, s * (1.0 - s), "sigmoid")

    # -- backpropagation --------------------------------------------------

    def backward(self) -> None:
        """Populate ``grad`` on every node that this value depends on."""
        # Iterative post-order DFS (avoids Python's recursion limit on deep
        # graphs). A node is appended only after all of its inputs, so the
        # reversed list visits each node before any of its inputs.
        topo: list[Value] = []
        visited: set[int] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            node, inputs_done = stack.pop()
            if inputs_done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((child, False) for child in node._prev if id(child) not in visited)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
