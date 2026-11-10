"""Gradient checks for every Tensor op against central finite differences."""

import numpy as np
import pytest

from mlfs.autograd import Tensor, cross_entropy, unbroadcast
from tests.gradcheck import numerical_grad

RNG = np.random.default_rng(0)


def check_grads(f, *arrays, rtol=1e-5, atol=1e-7):
    """Compare autograd gradients of scalar f(*tensors) to finite differences.

    To exercise non-trivial upstream gradients, f's output is contracted with a
    fixed random weight before differentiating.
    """
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = f(*tensors)
    weight = np.random.default_rng(42).normal(size=out.shape)
    (out * weight).sum().backward()

    for i, (t, a) in enumerate(zip(tensors, arrays, strict=True)):

        def scalar_f(x, i=i):
            args = [Tensor(b) for b in arrays]
            args[i] = Tensor(x)
            return float((f(*args).data * weight).sum())

        expected = numerical_grad(scalar_f, a)
        assert t.grad is not None, f"input {i} got no gradient"
        assert t.grad.shape == a.shape
        np.testing.assert_allclose(t.grad, expected, rtol=rtol, atol=atol, err_msg=f"input {i}")


def r(*shape):
    return RNG.normal(size=shape)


def positive(*shape):
    return RNG.uniform(0.5, 2.0, size=shape)


def away_from_zero(*shape):
    # ReLU is not differentiable at 0; keep finite differences off the kink.
    x = r(*shape)
    return np.where(np.abs(x) < 0.1, 0.5, x)


@pytest.mark.parametrize(
    "shapes",
    [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 1)), ((3, 4), ())],
)
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_broadcasting_binary_ops(op, shapes):
    a_shape, b_shape = shapes
    a = r(*a_shape)
    b = positive(*b_shape) if op == "div" else r(*b_shape)
    fn = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
    }[op]
    check_grads(fn, a, b)


@pytest.mark.parametrize(
    "a_shape,b_shape", [((3, 4), (4, 5)), ((2, 3, 4), (4, 5)), ((2, 3, 4), (2, 4, 2))]
)
def test_matmul(a_shape, b_shape):
    check_grads(lambda x, y: x @ y, r(*a_shape), r(*b_shape))


def test_matmul_rejects_vectors():
    with pytest.raises(ValueError):
        Tensor(r(3)) @ Tensor(r(3, 2))


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True), ((0, 2), False)])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_reductions(reduction, axis, keepdims):
    check_grads(lambda x: getattr(x, reduction)(axis=axis, keepdims=keepdims), r(2, 3, 4))


@pytest.mark.parametrize(
    "name,fn,make",
    [
        ("relu", lambda x: x.relu(), away_from_zero),
        ("tanh", lambda x: x.tanh(), r),
        ("exp", lambda x: x.exp(), r),
        ("log", lambda x: x.log(), positive),
        ("sigmoid", lambda x: x.sigmoid(), r),
        ("pow", lambda x: x**3, r),
        ("neg_pow", lambda x: x**-0.5, positive),
        ("log_softmax", lambda x: x.log_softmax(axis=-1), r),
        ("log_softmax_axis0", lambda x: x.log_softmax(axis=0), r),
        ("transpose", lambda x: x.T, r),
        ("reshape", lambda x: x.reshape(4, 3), r),
        ("getitem", lambda x: x[np.array([0, 2, 0]), 1:], r),
        ("rsub_rdiv", lambda x: 2.0 - x + 1.0 / (x * x + 1.0), r),
    ],
)
def test_unary_ops(name, fn, make):
    check_grads(fn, make(3, 4))


def test_cross_entropy_gradient_and_value():
    logits = r(5, 4)
    targets = np.array([0, 3, 1, 1, 2])
    check_grads(lambda z: cross_entropy(z, targets), logits)
    # Agrees with the unfused composition -mean(log_softmax[range, y]).
    composed = -Tensor(logits).log_softmax()[np.arange(5), targets].mean()
    assert cross_entropy(Tensor(logits), targets).item() == pytest.approx(composed.item())


def test_cross_entropy_is_stable_for_huge_logits():
    logits = Tensor(np.array([[1000.0, 0.0], [0.0, 1000.0]]), requires_grad=True)
    loss = cross_entropy(logits, np.array([0, 1]))
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert np.all(np.isfinite(logits.grad))


def test_two_layer_network_end_to_end():
    x, y = r(6, 5), np.array([0, 1, 2, 1, 0, 2])

    def net(w1, b1, w2):
        return cross_entropy((x @ w1 + b1).tanh() @ w2, y)

    check_grads(net, r(5, 4), r(4), r(4, 3))


def test_gradient_accumulates_over_reuse():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    (x * x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)


def test_only_leaves_keep_grad_and_constants_are_untracked():
    w = Tensor(r(3, 2), requires_grad=True)
    c = Tensor(r(4, 3))
    hidden = c @ w
    hidden.sum().backward()
    assert c.grad is None and hidden.grad is None
    assert w.grad.shape == (3, 2)


def test_backward_requires_scalar_or_explicit_grad():
    t = Tensor(r(2, 2), requires_grad=True)
    with pytest.raises(RuntimeError):
        (t * 2.0).backward()
    with pytest.raises(RuntimeError):
        Tensor(1.0).backward()


def test_float32_stays_float32():
    w = Tensor(np.ones((3, 2), dtype=np.float32), requires_grad=True)
    out = (Tensor(np.ones((4, 3), dtype=np.float32)) @ w * 0.5).mean()
    out.backward()
    assert out.data.dtype == np.float32
    assert w.grad.dtype == np.float32


def test_unbroadcast():
    g = np.ones((2, 3, 4))
    assert unbroadcast(g, (4,)).shape == (4,)
    np.testing.assert_allclose(unbroadcast(g, (3, 1)), np.full((3, 1), 8.0))
