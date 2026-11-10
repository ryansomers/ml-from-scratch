import numpy as np
import pytest

from mlfs.logistic_regression import LogisticRegression, SoftmaxRegression, sigmoid


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    logits = X @ np.array([2.0, -1.0, 0.5, 0.0]) + 0.3
    y = (rng.random(300) < sigmoid(logits)).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [3, 0], [0, 3]])
    X = np.vstack([c + rng.normal(size=(60, 2)) for c in centers])
    y = np.repeat(np.array(["a", "b", "c"]), 60)
    return X, y


def test_sigmoid_is_stable_for_extreme_inputs():
    z = np.array([-1000.0, 0.0, 1000.0])
    np.testing.assert_allclose(sigmoid(z), [0.0, 0.5, 1.0])


def test_gd_and_newton_agree(binary_data):
    X, y = binary_data
    gd = LogisticRegression(lam=0.01, solver="gd", tol=1e-9, max_iter=50_000).fit(X, y)
    newton = LogisticRegression(lam=0.01, solver="newton").fit(X, y)
    np.testing.assert_allclose(gd.coef_, newton.coef_, atol=1e-6)
    assert gd.intercept_ == pytest.approx(newton.intercept_, abs=1e-6)
    assert newton.n_iter_ < 15  # Newton converges quadratically
    assert newton.score(X, y) > 0.75


@pytest.mark.parametrize("C", [0.1, 1.0])
def test_matches_sklearn_with_equivalent_regularisation(binary_data, C):
    from sklearn.linear_model import LogisticRegression as SkLR

    X, y = binary_data
    ours = LogisticRegression(lam=1.0 / (C * len(X)), solver="newton").fit(X, y)
    theirs = SkLR(C=C, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.coef_, theirs.coef_.ravel(), atol=1e-5)
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-6)


def test_l2_shrinks_weights(binary_data):
    X, y = binary_data
    weak = LogisticRegression(lam=1e-4).fit(X, y)
    strong = LogisticRegression(lam=1.0).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)


def test_gd_loss_is_monotone(binary_data):
    X, y = binary_data
    model = LogisticRegression(lam=0.1, solver="gd", max_iter=200).fit(X, y)
    assert np.all(np.diff(model.loss_history_) <= 1e-12)


def test_binary_rejects_multiclass(multiclass_data):
    with pytest.raises(ValueError):
        LogisticRegression().fit(*multiclass_data)


def test_softmax_learns_blobs_and_returns_original_labels(multiclass_data):
    X, y = multiclass_data
    model = SoftmaxRegression(lam=0.01).fit(X, y)
    assert model.score(X, y) > 0.9
    assert set(model.predict(X)) <= {"a", "b", "c"}
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_softmax_matches_sklearn_multinomial(multiclass_data):
    from sklearn.linear_model import LogisticRegression as SkLR

    X, y = multiclass_data
    C = 1.0
    ours = SoftmaxRegression(lam=1.0 / (C * len(X)), tol=1e-9, max_iter=50_000).fit(X, y)
    theirs = SkLR(C=C, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-5)
