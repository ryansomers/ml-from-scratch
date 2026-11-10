import numpy as np
import pytest

from mlfs.pca import PCA


@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(300, 2)) * np.array([5.0, 1.0])
    mixing = rng.normal(size=(2, 6))
    return latent @ mixing + 0.05 * rng.normal(size=(300, 6))


def test_full_rank_roundtrip_is_exact(correlated_data):
    pca = PCA().fit(correlated_data)
    Z = pca.transform(correlated_data)
    np.testing.assert_allclose(pca.inverse_transform(Z), correlated_data, atol=1e-10)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)


def test_components_are_orthonormal_and_scores_uncorrelated(correlated_data):
    pca = PCA(n_components=3).fit(correlated_data)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(3), atol=1e-12)
    Z = pca.transform(correlated_data)
    cov = np.cov(Z, rowvar=False)
    np.testing.assert_allclose(cov, np.diag(pca.explained_variance_), atol=1e-8)


def test_two_latent_factors_explain_almost_everything(correlated_data):
    pca = PCA(n_components=0.999).fit(correlated_data)
    assert pca.n_components_ == 2


def test_matches_sklearn(correlated_data):
    from sklearn.decomposition import PCA as SkPCA

    ours = PCA(n_components=4).fit(correlated_data)
    theirs = SkPCA(n_components=4, svd_solver="full").fit(correlated_data)
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, theirs.explained_variance_ratio_, rtol=1e-10
    )
    np.testing.assert_allclose(ours.explained_variance_, theirs.explained_variance_, rtol=1e-10)
    # Components agree up to sign.
    np.testing.assert_allclose(np.abs(ours.components_), np.abs(theirs.components_), atol=1e-8)


def test_invalid_n_components(correlated_data):
    with pytest.raises(ValueError):
        PCA(n_components=0).fit(correlated_data)
    with pytest.raises(ValueError):
        PCA(n_components=99).fit(correlated_data)
