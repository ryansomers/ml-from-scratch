import numpy as np
import pytest

from mlfs.autograd import MLP, SGD, Adam, Linear, Tensor, cross_entropy


def quadratic_bowl():
    """f(p) = sum((p - target)^2), minimised at target."""
    target = np.array([1.0, -2.0, 3.0])
    p = Tensor(np.zeros(3), requires_grad=True)
    return p, target, lambda: ((p - target) ** 2).sum()


@pytest.mark.parametrize(
    "make_opt",
    [
        lambda ps: SGD(ps, lr=0.1),
        lambda ps: SGD(ps, lr=0.05, momentum=0.9),
        lambda ps: Adam(ps, lr=0.1),
    ],
)
def test_optimizers_minimise_quadratic(make_opt):
    p, target, loss_fn = quadratic_bowl()
    opt = make_opt([p])
    for _ in range(500):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    np.testing.assert_allclose(p.data, target, atol=1e-3)


def test_sgd_momentum_matches_manual_update():
    p = Tensor(np.array([1.0]), requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = np.array([2.0])
    opt.step()  # v = 2, p = 1 - 0.2
    p.grad = np.array([2.0])
    opt.step()  # v = 0.9*2 + 2 = 3.8, p = 0.8 - 0.38
    np.testing.assert_allclose(p.data, [0.42])


def test_adam_first_step_has_magnitude_lr():
    # After bias correction, m_hat = g and v_hat = g^2, so the first step is
    # lr * g / |g| = lr * sign(g) regardless of the gradient's scale.
    p = Tensor(np.array([0.0, 0.0]), requires_grad=True)
    opt = Adam([p], lr=0.01, eps=0.0)
    p.grad = np.array([1000.0, -0.001])
    opt.step()
    np.testing.assert_allclose(p.data, [-0.01, 0.01])


def test_adam_matches_reference_formula_over_steps():
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(5, 3))
    p = Tensor(np.zeros(3), requires_grad=True)
    opt = Adam([p], lr=0.1, betas=(0.8, 0.9), eps=1e-8)
    m = v = np.zeros(3)
    expected = np.zeros(3)
    for t, g in enumerate(grads, start=1):
        p.grad = g
        opt.step()
        m = 0.8 * m + 0.2 * g
        v = 0.9 * v + 0.1 * g**2
        expected = expected - 0.1 * (m / (1 - 0.8**t)) / (np.sqrt(v / (1 - 0.9**t)) + 1e-8)
    np.testing.assert_allclose(p.data, expected)


def test_weight_decay_shrinks_parameters():
    p = Tensor(np.array([5.0]), requires_grad=True)
    opt = SGD([p], lr=0.1, weight_decay=1.0)
    p.grad = np.array([0.0])
    opt.step()
    np.testing.assert_allclose(p.data, [4.5])


def test_module_collects_parameters():
    model = MLP([4, 8, 3], rng=np.random.default_rng(0))
    shapes = [p.shape for p in model.parameters()]
    assert shapes == [(4, 8), (8,), (8, 3), (3,)]
    assert len(Linear(2, 2, bias=False).parameters()) == 1


def test_mlp_learns_nonlinear_spirals():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0.2, 1, n)
    X = np.vstack(
        [
            np.column_stack([t * np.cos(4 * t + k * np.pi), t * np.sin(4 * t + k * np.pi)])
            for k in range(2)
        ]
    )
    y = np.repeat([0, 1], n)
    model = MLP([2, 32, 32, 2], rng=rng)
    opt = Adam(model.parameters(), lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        loss = cross_entropy(model(Tensor(X)), y)
        loss.backward()
        opt.step()
    accuracy = np.mean(model(Tensor(X)).data.argmax(axis=1) == y)
    assert accuracy > 0.95
    assert loss.item() < 0.2


def test_optimizer_needs_params():
    with pytest.raises(ValueError):
        SGD([], lr=0.1)
