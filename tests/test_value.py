import math

import numpy as np
import pytest

from mlfs.autograd import Value
from tests.gradcheck import numerical_grad


def _dispatch(method: str, fallback):
    """Call ``x.<method>()`` on a Value, or the plain-float ``fallback`` otherwise."""
    return lambda x: getattr(x, method)() if isinstance(x, Value) else fallback(x)


tanh = _dispatch("tanh", math.tanh)
relu = _dispatch("relu", lambda x: max(x, 0.0))
exp = _dispatch("exp", math.exp)
log = _dispatch("log", math.log)
sigmoid = _dispatch("sigmoid", lambda x: 1.0 / (1.0 + math.exp(-x)))

# Each case is a scalar function of two inputs, written once and evaluated
# both on Values (for autograd) and on floats (for finite differences).
CASES = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "rsub_rdiv": lambda a, b: 3.0 - a + 2.0 / b,
    "pow": lambda a, b: a**3 + b**-2,
    "tanh": lambda a, b: tanh(a * b),
    "relu": lambda a, b: relu(a - b),
    "exp": lambda a, b: exp(a + b),
    "log": lambda a, b: log(a * a + b),
    "sigmoid": lambda a, b: sigmoid(a * b),
    "reused_node": lambda a, b: a * a * b + a,
}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("point", [(0.7, 1.3), (-1.2, 2.5)])
def test_value_gradients_match_finite_differences(name, point):
    f = CASES[name]
    a, b = Value(point[0]), Value(point[1])
    out = f(a, b)
    out.backward()
    expected = numerical_grad(lambda x: f(float(x[0]), float(x[1])), np.array(point))
    assert out.data == pytest.approx(f(*point))
    np.testing.assert_allclose([a.grad, b.grad], expected, rtol=1e-6, atol=1e-8)


def test_small_neuron_matches_hand_derivation():
    # o = tanh(w1 x1 + w2 x2 + b); do/dw1 = (1 - o^2) x1.
    x1, x2, w1, w2, b = Value(2.0), Value(0.0), Value(-3.0), Value(1.0), Value(6.8813735870)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()
    assert o.data == pytest.approx(0.7071, abs=1e-4)
    assert w1.grad == pytest.approx((1 - o.data**2) * 2.0)
    assert x2.grad == pytest.approx((1 - o.data**2) * 1.0)


def test_deep_chain_does_not_hit_recursion_limit():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.0
    y.backward()
    assert x.grad == 1.0


def test_value_exponent_rejected():
    with pytest.raises(TypeError):
        _ = Value(2.0) ** Value(3.0)
