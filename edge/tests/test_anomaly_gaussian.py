import numpy as np
import pytest

from e3_edge.anomaly.gaussian import EmbeddingModel, GaussianModel, RunningMoments, oas_shrink


def correlated(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    return rng.normal(size=(n, dim)) @ mixing.T + rng.normal(size=dim)


def test_streaming_moments_match_numpy_whatever_the_batching():
    x = correlated(1000, 12)
    moments = RunningMoments(12, batch_size=64)
    for row in x[:333]:
        moments.add(row)
    moments.add_batch(x[333:900])
    for row in x[900:]:
        moments.add(row)
    assert moments.n == 1000
    np.testing.assert_allclose(moments.mean, x.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(moments.covariance(), np.cov(x, rowvar=False, bias=True), atol=1e-10)


def test_moments_state_round_trip_continues_exactly():
    x = correlated(500, 6)
    a = RunningMoments(6)
    a.add_batch(x[:200])
    b = RunningMoments.from_state(a.state())
    a.add_batch(x[200:])
    b.add_batch(x[200:])
    np.testing.assert_allclose(a.covariance(), b.covariance())


def test_oas_matches_scikit_learn():
    sklearn_cov = pytest.importorskip("sklearn.covariance")
    x = correlated(60, 40, seed=3)  # few samples for the dimension: shrinkage matters
    shrunk, shrinkage = oas_shrink(np.cov(x, rowvar=False, bias=True), len(x))
    ref = sklearn_cov.OAS().fit(x)
    assert shrinkage == pytest.approx(ref.shrinkage_)
    np.testing.assert_allclose(shrunk, ref.covariance_, atol=1e-12)


def test_full_model_distance_is_the_mahalanobis_formula():
    x = correlated(2000, 10)
    moments = RunningMoments(10)
    moments.add_batch(x)
    model = GaussianModel.fit(moments, ridge=0.0)
    cov, _ = oas_shrink(moments.covariance(), moments.n)
    point = x[0] + 3.0
    d = point - moments.mean
    expected = np.sqrt(d @ np.linalg.solve(cov, d))
    assert model.distance(point) == pytest.approx(expected, rel=1e-4)


def test_reduced_model_is_exact_mahalanobis_under_its_ppca_covariance():
    x = correlated(3000, 20, seed=1)
    moments = RunningMoments(20)
    moments.add_batch(x)
    model = GaussianModel.fit(moments, n_components=5)
    assert model.reduced and len(model.eigvals) == 5
    points = x[:7] * 1.5
    d = points - model.mean
    expected = np.sqrt(np.einsum("ij,ij->i", d @ np.linalg.inv(model.covariance()), d))
    np.testing.assert_allclose(model.distance(points), expected, rtol=1e-3)


def test_batch_and_single_scoring_agree():
    x = correlated(800, 8)
    moments = RunningMoments(8)
    moments.add_batch(x)
    model = GaussianModel.fit(moments)
    batch = model.distance(x[:20])
    single = [float(model.distance(row)) for row in x[:20]]
    np.testing.assert_allclose(batch, single, rtol=1e-5)


def test_dead_dimensions_do_not_break_the_model():
    # YAMNet embeddings have dimensions that are almost always zero: covariance is singular.
    x = correlated(500, 16)
    x[:, :5] = 0.0
    moments = RunningMoments(16)
    moments.add_batch(x)
    model = GaussianModel.fit(moments)
    d = model.distance(np.vstack([x[:10], x[:10] + 1.0]))
    assert np.all(np.isfinite(d))
    assert d[10:].mean() > d[:10].mean()


def test_squared_distance_of_gaussian_data_averages_the_dimension():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20000, 16))
    moments = RunningMoments(16)
    moments.add_batch(x)
    model = GaussianModel.fit(moments)
    assert model.distance_squared(rng.normal(size=(5000, 16))).mean() == pytest.approx(16, rel=0.05)


def test_blend_weights():
    x = correlated(1000, 6)
    moments = RunningMoments(6)
    moments.add_batch(x)
    old = GaussianModel.fit(moments, ridge=0.0)
    new_mean, new_cov = old.mean + 2.0, old.covariance() * 2
    same = old.blend(new_mean, new_cov, 0.0, 10, ridge=0.0)
    np.testing.assert_allclose(same.mean, old.mean, atol=1e-5)
    np.testing.assert_allclose(same.covariance(), old.covariance(), rtol=1e-4, atol=1e-6)
    half = old.blend(new_mean, new_cov, 0.5, 10, ridge=0.0)
    np.testing.assert_allclose(half.mean, old.mean + 1.0, atol=1e-5)
    delta = np.full(6, 2.0)
    expected = 0.5 * old.covariance() + 0.5 * new_cov + 0.25 * np.outer(delta, delta)
    np.testing.assert_allclose(half.covariance(), expected, rtol=1e-4, atol=1e-6)


def test_selective_model_takes_the_closest_component():
    a, b = correlated(1000, 4, seed=1), correlated(1000, 4, seed=2) + 10
    comps = {}
    for name, data in (("global", np.vstack([a, b])), ("a", a), ("b", b)):
        m = RunningMoments(4)
        m.add_batch(data)
        comps[name] = GaussianModel.fit(m)
    model = EmbeddingModel(comps, "category")
    point = b[0]
    distance, name = model.score(point)
    assert name == "b"
    assert distance == pytest.approx(min(float(c.distance(point)) for c in comps.values()))
    np.testing.assert_allclose(model.score_batch(b[:5]), [model.score(p)[0] for p in b[:5]], rtol=1e-5)
    assert EmbeddingModel(comps, "global").score(point)[1] == "global"


def test_model_array_round_trip():
    moments = RunningMoments(5)
    moments.add_batch(correlated(300, 5))
    model = GaussianModel.fit(moments, n_components=3)
    again = GaussianModel.from_arrays(model.arrays())
    point = np.ones(5)
    assert float(again.distance(point)) == pytest.approx(float(model.distance(point)))
