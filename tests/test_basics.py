import numpy as np

from mlfs.metrics import accuracy_score, r2_score
from mlfs.preprocessing import StandardScaler


def test_accuracy_and_r2():
    assert accuracy_score(np.array([0, 1, 1]), np.array([0, 1, 0])) == 2 / 3
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == 1.0
    assert r2_score(y, np.full(3, 2.0)) == 0.0


def test_standard_scaler_roundtrip_and_constant_column():
    X = np.column_stack([np.arange(5.0), np.full(5, 7.0)])
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    np.testing.assert_allclose(Z[:, 0].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(Z[:, 0].std(), 1)
    np.testing.assert_allclose(Z[:, 1], 0)
    np.testing.assert_allclose(scaler.inverse_transform(Z), X)
