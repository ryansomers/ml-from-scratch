import numpy as np
import pytest

from mlfs.naive_bayes import GaussianNB, MultinomialNB


@pytest.fixture
def gaussian_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(80, 3)), rng.normal(2, 0.5, size=(40, 3))])
    y = np.array([0] * 80 + [1] * 40)
    return X, y


@pytest.fixture
def count_data():
    rng = np.random.default_rng(1)
    probs = np.array([[0.6, 0.3, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4]])
    X = np.vstack([rng.multinomial(20, p, size=50) for p in probs])
    y = np.repeat(["spam", "ham"], 50)
    return X, y


def test_gaussian_nb_parameters(gaussian_data):
    X, y = gaussian_data
    model = GaussianNB().fit(X, y)
    np.testing.assert_allclose(model.theta_[1], X[y == 1].mean(axis=0))
    np.testing.assert_allclose(np.exp(model.class_log_prior_), [80 / 120, 40 / 120])
    assert model.score(X, y) > 0.9


def test_gaussian_nb_matches_sklearn(gaussian_data):
    from sklearn.naive_bayes import GaussianNB as SkGNB

    X, y = gaussian_data
    ours, theirs = GaussianNB().fit(X, y), SkGNB().fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-10)


def test_gaussian_nb_handles_constant_feature():
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 8.0], [0.0, 9.0]])
    model = GaussianNB().fit(X, [0, 0, 1, 1])
    assert np.all(np.isfinite(model.predict_log_proba(X)))


def test_multinomial_nb_matches_sklearn(count_data):
    from sklearn.naive_bayes import MultinomialNB as SkMNB

    X, y = count_data
    ours, theirs = MultinomialNB(alpha=0.5).fit(X, y), SkMNB(alpha=0.5).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-10)
    assert ours.score(X, y) > 0.95


def test_multinomial_nb_rejects_negative_counts():
    with pytest.raises(ValueError):
        MultinomialNB().fit(np.array([[-1.0, 2.0]]), [0])
