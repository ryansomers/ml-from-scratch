import numpy as np
import pytest

from mlfs.kmeans import KMeans, kmeans_plus_plus, squared_distances


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([c + rng.normal(size=(50, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 50)
    return X, y


def _same_partition(a, b):
    # Labels are arbitrary; two partitions match if the label pairs are 1:1.
    pairs = set(zip(a.tolist(), b.tolist(), strict=True))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_squared_distances_matches_brute_force():
    rng = np.random.default_rng(1)
    X, C = rng.normal(size=(7, 3)), rng.normal(size=(4, 3))
    brute = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(squared_distances(X, C), brute, atol=1e-12)


def test_recovers_well_separated_blobs(blobs):
    X, y = blobs
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    assert _same_partition(model.labels_, y)
    np.testing.assert_array_equal(model.predict(X), model.labels_)
    expected = sum(((X[y == j] - X[y == j].mean(0)) ** 2).sum() for j in range(3))
    assert model.inertia_ == pytest.approx(expected)


def test_kmeans_plus_plus_usually_seeds_every_blob(blobs):
    X, y = blobs

    def covers_all_blobs(seed):
        centers = kmeans_plus_plus(X, 3, np.random.default_rng(seed))
        hit = {y[np.argmin(((X - c) ** 2).sum(1))] for c in centers}
        return hit == {0, 1, 2}

    # D^2 sampling is random, so test the rate over many seeds. Uniform random
    # seeding would cover all three blobs only 2/9 of the time.
    rate = np.mean([covers_all_blobs(seed) for seed in range(100)])
    assert rate > 0.8


def test_more_inits_never_worse(blobs):
    X, _ = blobs
    one = KMeans(n_clusters=5, n_init=1, init="random", random_state=3).fit(X)
    many = KMeans(n_clusters=5, n_init=20, init="random", random_state=3).fit(X)
    assert many.inertia_ <= one.inertia_ + 1e-9


def test_seeded_runs_are_reproducible(blobs):
    X, _ = blobs
    a = KMeans(n_clusters=4, random_state=7).fit(X)
    b = KMeans(n_clusters=4, random_state=7).fit(X)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)


def test_inertia_close_to_sklearn(blobs):
    from sklearn.cluster import KMeans as SkKMeans

    X, _ = blobs
    ours = KMeans(n_clusters=3, random_state=0).fit(X)
    theirs = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ == pytest.approx(theirs.inertia_, rel=1e-6)
    assert ours.score(X) == pytest.approx(-ours.inertia_)


def test_duplicate_points_do_not_crash():
    X = np.ones((10, 2))
    model = KMeans(n_clusters=3, n_init=2, random_state=0).fit(X)
    assert model.inertia_ == 0.0
