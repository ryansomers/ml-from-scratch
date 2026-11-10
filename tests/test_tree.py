import numpy as np
import pytest

from mlfs.tree import DecisionTreeClassifier, RandomForestClassifier, entropy, gini


@pytest.fixture
def xor_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.8 * rng.normal(size=300) > 0).astype(int)
    return X[:200], y[:200], X[200:], y[200:]


def test_impurity_functions():
    counts = np.array([[5, 5], [10, 0], [0, 0]], dtype=float)
    np.testing.assert_allclose(gini(counts), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(entropy(counts), [1.0, 0.0, 0.0])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tree_fits_xor_perfectly(xor_data, criterion):
    X, y = xor_data
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    assert tree.score(X, y) == 1.0


def test_max_depth_is_respected(xor_data):
    X, y = xor_data
    for depth in (1, 2, 3):
        assert DecisionTreeClassifier(max_depth=depth).fit(X, y).depth_ <= depth
    stump = DecisionTreeClassifier(max_depth=0).fit(X, y)
    assert stump.depth_ == 0
    assert len(set(stump.predict(X))) == 1


def test_min_samples_split_limits_growth(noisy_data):
    X, y, _, _ = noisy_data
    full = DecisionTreeClassifier().fit(X, y)
    pruned = DecisionTreeClassifier(min_samples_split=50).fit(X, y)
    assert pruned.depth_ < full.depth_
    assert full.score(X, y) == 1.0


def test_min_samples_leaf():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.array([0] * 9 + [1])
    tree = DecisionTreeClassifier(min_samples_leaf=2).fit(X, y)
    # Isolating the single positive would leave a leaf of size 1.
    proba = tree.predict_proba(X)
    assert proba[-1, 1] < 1.0


def test_matches_sklearn_tree_on_unambiguous_data(noisy_data):
    from sklearn.tree import DecisionTreeClassifier as SkTree

    X, y, X_test, _ = noisy_data
    ours = DecisionTreeClassifier(max_depth=3).fit(X, y)
    theirs = SkTree(max_depth=3, random_state=0).fit(X, y)
    np.testing.assert_array_equal(ours.predict(X_test), theirs.predict(X_test))
    np.testing.assert_allclose(ours.feature_importances_, theirs.feature_importances_, atol=1e-9)


def test_string_labels_and_proba(xor_data):
    X, y = xor_data
    labels = np.where(y == 1, "yes", "no")
    tree = DecisionTreeClassifier(max_depth=4).fit(X, labels)
    assert set(tree.predict(X)) == {"yes", "no"}
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1.0)


def test_forest_beats_single_tree_on_noisy_data(noisy_data):
    X, y, X_test, y_test = noisy_data
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    forest = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    assert forest.score(X_test, y_test) > tree.score(X_test, y_test)
    np.testing.assert_allclose(forest.predict_proba(X_test).sum(axis=1), 1.0)
    # The two informative features should dominate the importances.
    assert set(np.argsort(forest.feature_importances_)[-2:]) == {0, 1}


def test_forest_is_reproducible_and_trees_differ(noisy_data):
    X, y, X_test, _ = noisy_data
    a = RandomForestClassifier(n_estimators=5, random_state=3).fit(X, y)
    b = RandomForestClassifier(n_estimators=5, random_state=3).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X_test), b.predict_proba(X_test))
    roots = {(t.root_.feature, t.root_.threshold) for t in a.estimators_}
    assert len(roots) > 1


def test_invalid_params():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="mse")
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_features="half").fit(np.ones((4, 2)), [0, 1, 0, 1])
