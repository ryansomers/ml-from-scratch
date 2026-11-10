import numpy as np
import pytest

from mlfs.linear_regression import (
    LinearRegression,
    Ridge,
    back_substitution,
    householder_qr,
)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    true_w = np.array([1.5, -2.0, 0.0, 3.0, 0.5])
    y = X @ true_w + 4.0 + 0.1 * rng.normal(size=200)
    return X, y, true_w


@pytest.mark.parametrize("shape", [(6, 6), (20, 5), (50, 1), (8, 3)])
def test_householder_qr_reconstructs_and_is_orthonormal(shape):
    A = np.random.default_rng(1).normal(size=shape)
    Q, R = householder_qr(A)
    n = shape[1]
    np.testing.assert_allclose(Q @ R, A, atol=1e-12)
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-12)
    assert np.allclose(R, np.triu(R))


def test_householder_qr_matches_numpy_up_to_signs():
    A = np.random.default_rng(2).normal(size=(30, 4))
    Q, R = householder_qr(A)
    Q_np, R_np = np.linalg.qr(A)
    # QR is unique only up to the sign of each column of Q / row of R.
    signs = np.sign(np.diag(R)) * np.sign(np.diag(R_np))
    np.testing.assert_allclose(Q * signs, Q_np, atol=1e-12)
    np.testing.assert_allclose(R * signs[:, None], R_np, atol=1e-12)


def test_householder_qr_rejects_wide_matrix():
    with pytest.raises(ValueError):
        householder_qr(np.ones((2, 3)))


def test_back_substitution():
    R = np.triu(np.random.default_rng(3).normal(size=(5, 5))) + 5 * np.eye(5)
    b = np.arange(5.0)
    np.testing.assert_allclose(R @ back_substitution(R, b), b, atol=1e-12)


def test_all_solvers_agree_on_coefficients(regression_data):
    X, y, true_w = regression_data
    fits = {s: LinearRegression(solver=s).fit(X, y) for s in ("normal", "qr", "gd")}
    reference = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)[0]
    for solver, model in fits.items():
        np.testing.assert_allclose(model.coef_, reference[:-1], atol=1e-6, err_msg=solver)
        assert model.intercept_ == pytest.approx(reference[-1], abs=1e-6)
    np.testing.assert_allclose(fits["normal"].coef_, true_w, atol=0.05)
    assert fits["qr"].score(X, y) > 0.99


def test_gradient_descent_loss_decreases(regression_data):
    X, y, _ = regression_data
    model = LinearRegression(solver="gd").fit(X, y)
    losses = np.array(model.loss_history_)
    assert np.all(np.diff(losses) <= 1e-12)


@pytest.mark.parametrize("solver", ["normal", "qr", "gd"])
def test_ridge_matches_closed_form_and_shrinks(regression_data, solver):
    X, y, _ = regression_data
    alpha = 10.0
    model = Ridge(alpha=alpha, solver=solver).fit(X, y)
    Xc, yc = X - X.mean(0), y - y.mean()
    expected = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(5), Xc.T @ yc)
    np.testing.assert_allclose(model.coef_, expected, atol=1e-6)
    ols = LinearRegression(solver=solver).fit(X, y)
    assert np.linalg.norm(model.coef_) < np.linalg.norm(ols.coef_)


def test_ridge_matches_sklearn(regression_data):
    from sklearn.linear_model import Ridge as SkRidge

    X, y, _ = regression_data
    ours = Ridge(alpha=3.0, solver="qr").fit(X, y)
    theirs = SkRidge(alpha=3.0).fit(X, y)
    np.testing.assert_allclose(ours.coef_, theirs.coef_, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)


def test_no_intercept():
    X = np.random.default_rng(4).normal(size=(50, 2))
    y = X @ np.array([2.0, -1.0])
    model = LinearRegression(solver="qr", fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(model.coef_, [2.0, -1.0], atol=1e-10)
    assert model.intercept_ == 0.0


def test_invalid_solver():
    with pytest.raises(ValueError):
        Ridge(solver="magic")
