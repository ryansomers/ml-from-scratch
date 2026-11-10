"""Vectorised reverse-mode autograd over NumPy arrays.

Design
------
Each operation creates an output ``Tensor`` that stores its parent tensors and
a ``_backward`` function mapping the upstream gradient dL/d(out) to one
gradient per parent (the vector-Jacobian product). ``Tensor.backward`` walks
the graph in reverse topological order and lets the engine, not each op,
handle two pieces of bookkeeping:

* Broadcasting. If a parent of shape (4,) was broadcast to (3, 4) in the
  forward pass, its gradient must be summed over the broadcast axis.
  ``unbroadcast`` does this for every op, so individual ops can return
  gradients in the output's shape.
* Accumulation. A tensor used several times receives the sum of the
  gradients from each use.

Only tensors with ``requires_grad=True`` (or that depend on one) are tracked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting.

    Broadcasting (a) prepends size-1 axes and (b) stretches size-1 axes. The
    adjoint of copying a value along an axis is summing along it, so we sum
    away the prepended axes and sum (keeping dims) over stretched ones.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    # Make ``ndarray + Tensor`` dispatch to Tensor.__radd__ instead of NumPy
    # trying to treat the Tensor as an element.
    __array_priority__ = 100.0

    def __init__(
        self,
        data: np.ndarray | float | int | Sequence,
        requires_grad: bool = False,
        _parents: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        arr = np.asarray(data)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        self.data: np.ndarray = arr
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents = _parents
        self._op = _op
        self._backward: Callable[[np.ndarray], tuple[np.ndarray | None, ...]] | None = None

    # -- basics -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> Tensor:
        return Tensor(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    def _wrap(self, x: Tensor | np.ndarray | float) -> Tensor:
        """Turn a constant operand into a Tensor (that does not require grad).

        Python scalars adopt this tensor's dtype, so ``x * 0.5`` on a float32
        tensor stays float32 instead of being promoted to float64.
        """
        if isinstance(x, Tensor):
            return x
        if isinstance(x, np.ndarray):
            return Tensor(x)
        return Tensor(np.asarray(x, dtype=self.data.dtype))

    def _make(
        self,
        data: np.ndarray,
        parents: tuple[Tensor, ...],
        backward: Callable[[np.ndarray], tuple[np.ndarray | None, ...]],
        op: str,
    ) -> Tensor:
        """Create an op's output, recording the graph only if a parent needs grads."""
        needs_grad = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs_grad, _parents=parents if needs_grad else (), _op=op)
        if needs_grad:
            out._backward = backward
        return out

    # -- elementwise arithmetic (broadcasting) ------------------------------

    def __add__(self, other: Tensor | np.ndarray | float) -> Tensor:
        other = self._wrap(other)
        return self._make(self.data + other.data, (self, other), lambda g: (g, g), "add")

    def __mul__(self, other: Tensor | np.ndarray | float) -> Tensor:
        other = self._wrap(other)
        a, b = self.data, other.data
        return self._make(a * b, (self, other), lambda g: (g * b, g * a), "mul")

    def __truediv__(self, other: Tensor | np.ndarray | float) -> Tensor:
        other = self._wrap(other)
        a, b = self.data, other.data
        return self._make(a / b, (self, other), lambda g: (g / b, -g * a / (b * b)), "div")

    def __neg__(self) -> Tensor:
        return self._make(-self.data, (self,), lambda g: (-g,), "neg")

    def __sub__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return self + (-self._wrap(other))

    def __pow__(self, p: float) -> Tensor:
        """Elementwise power with a constant exponent."""
        if isinstance(p, Tensor):
            raise TypeError("only constant exponents are supported")
        a = self.data
        return self._make(a**p, (self,), lambda g: (g * p * a ** (p - 1),), f"pow{p}")

    def __radd__(self, other: np.ndarray | float) -> Tensor:
        return self + other

    def __rmul__(self, other: np.ndarray | float) -> Tensor:
        return self * other

    def __rsub__(self, other: np.ndarray | float) -> Tensor:
        return self._wrap(other) - self

    def __rtruediv__(self, other: np.ndarray | float) -> Tensor:
        return self._wrap(other) / self

    # -- linear algebra -----------------------------------------------------

    def __matmul__(self, other: Tensor | np.ndarray) -> Tensor:
        """Matrix product; both operands must be at least 2-D.

        For C = A B and upstream G = dL/dC: dL/dA = G B^T and dL/dB = A^T G.
        Leading (batch) axes broadcast like np.matmul and are summed back by
        ``unbroadcast``.
        """
        other = self._wrap(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul needs operands with ndim >= 2; reshape vectors first")

        def backward(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return g @ np.swapaxes(b, -1, -2), np.swapaxes(a, -1, -2) @ g

        return self._make(a @ b, (self, other), backward, "matmul")

    def __rmatmul__(self, other: np.ndarray) -> Tensor:
        return self._wrap(other) @ self

    @property
    def T(self) -> Tensor:
        """Transpose of the last two axes."""
        return self._make(
            np.swapaxes(self.data, -1, -2),
            (self,),
            lambda g: (np.swapaxes(g, -1, -2),),
            "transpose",
        )

    def reshape(self, *shape: int) -> Tensor:
        original = self.shape
        return self._make(
            self.data.reshape(*shape), (self,), lambda g: (g.reshape(original),), "reshape"
        )

    def __getitem__(self, index) -> Tensor:
        original = self.data

        def backward(g: np.ndarray) -> tuple[np.ndarray]:
            grad = np.zeros_like(original)
            # add.at (not assignment) so repeated indices accumulate.
            np.add.at(grad, index, g)
            return (grad,)

        return self._make(self.data[index], (self,), backward, "getitem")

    # -- reductions ---------------------------------------------------------

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        shape = self.shape

        def backward(g: np.ndarray) -> tuple[np.ndarray]:
            # Re-insert reduced axes so g broadcasts back over the input.
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, shape),)

        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), backward, "sum")

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        total = self.sum(axis=axis, keepdims=keepdims)
        return total * (total.data.size / self.data.size)

    # -- elementwise nonlinearities ----------------------------------------

    def relu(self) -> Tensor:
        mask = self.data > 0
        return self._make(self.data * mask, (self,), lambda g: (g * mask,), "relu")

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        return self._make(t, (self,), lambda g: (g * (1.0 - t * t),), "tanh")

    def exp(self) -> Tensor:
        e = np.exp(self.data)
        return self._make(e, (self,), lambda g: (g * e,), "exp")

    def log(self) -> Tensor:
        a = self.data
        return self._make(np.log(a), (self,), lambda g: (g / a,), "log")

    def sigmoid(self) -> Tensor:
        s = 0.5 * (1.0 + np.tanh(0.5 * self.data))  # overflow-free form
        return self._make(s, (self,), lambda g: (g * s * (1.0 - s),), "sigmoid")

    def log_softmax(self, axis: int = -1) -> Tensor:
        """y = x - logsumexp(x) along ``axis``.

        Jacobian-vector product: dL/dx = g - softmax(x) * sum(g) along axis.
        """
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        softmax = np.exp(out)

        def backward(g: np.ndarray) -> tuple[np.ndarray]:
            return (g - softmax * g.sum(axis=axis, keepdims=True),)

        return self._make(out, (self,), backward, "log_softmax")

    # -- backpropagation ----------------------------------------------------

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Accumulate dL/dt into ``t.grad`` for every tracked tensor t in the graph."""
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)

        # Iterative post-order DFS; see Value.backward for the same idea.
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, inputs_done = stack.pop()
            if inputs_done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if id(p) not in visited)

        # Upstream gradients for intermediate nodes live here, so that only
        # user-visible tensors (leaves and the output) end up with ``.grad``.
        grads: dict[int, np.ndarray] = {id(self): np.asarray(grad, dtype=self.data.dtype)}
        for node in reversed(topo):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if not node._parents:  # leaf
                node.grad = g if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._backward(g), strict=True):
                if pg is None or not parent.requires_grad:
                    continue
                pg = unbroadcast(pg, parent.shape)
                key = id(parent)
                grads[key] = grads[key] + pg if key in grads else pg


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy for integer class ``targets``.

    Fused into one op because the combined gradient is simple and stable:
    dL/dlogits = (softmax(logits) - onehot(targets)) / n.
    """
    targets = np.asarray(targets)
    n = len(targets)
    z = logits.data
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), targets].mean()

    def backward(g: np.ndarray) -> tuple[np.ndarray]:
        dz = np.exp(log_probs)
        dz[np.arange(n), targets] -= 1.0
        return (g * dz / n,)

    return logits._make(np.asarray(loss, dtype=z.dtype), (logits,), backward, "cross_entropy")
