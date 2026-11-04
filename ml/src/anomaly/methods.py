"""Frame-level anomaly scorers compared in the evaluation (Table 1.4: "evaluate alternative
methods"). Every scorer is fitted on calibration frames only (no labels) and returns higher
scores for more unusual frames.

The Mahalanobis variants run the edge code itself (e3_edge.anomaly), so what is evaluated is
exactly what ships to the Pi. The others are offline baselines from scikit-learn.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from e3_edge.anomaly import AnomalyConfig, BaselineCalibrator, FrameStream
from e3_edge.anomaly.calibration import split_calibration
from e3_edge.anomaly.frames import observed_seconds
from e3_edge.anomaly.gaussian import EmbeddingModel, GaussianModel, RunningMoments
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


def threshold_split(stream: FrameStream, cfg: AnomalyConfig) -> int:
    """Row where the threshold phase starts, by the device's rule (last whole day if >= 2 days)."""
    total = observed_seconds(stream.timestamps, cfg.frame_seconds, cfg.max_gap_seconds)
    stats, _ = split_calibration(total)
    return int(len(stream) * stats / total)


class Scorer:
    name = "scorer"
    edge_ready = False  # runs on the Pi today

    def fit(self, stream: FrameStream, frame_quantile: float, seed: int = 0) -> Scorer:
        started = time.perf_counter()
        split = threshold_split(stream, AnomalyConfig())
        self._fit(stream.slice(0, split), seed)
        held_out = self.score(stream.slice(split, len(stream)))
        self.threshold = float(np.quantile(held_out, frame_quantile))
        self.fit_seconds = time.perf_counter() - started
        return self

    def _fit(self, stream: FrameStream, seed: int) -> None:
        raise NotImplementedError

    def score(self, stream: FrameStream) -> np.ndarray:
        raise NotImplementedError

    def size_bytes(self) -> int | None:
        return None


class EdgeBaseline(Scorer):
    """The shipped calibrator and model (contexts: global / category / period)."""

    edge_ready = True

    def __init__(self, contexts: str, n_components: int | None = None, cfg: AnomalyConfig | None = None):
        self.contexts, self.n_components = contexts, n_components
        self.cfg = cfg or AnomalyConfig()
        self.name = f"edge_{contexts}" + (f"_pca{n_components}" if n_components else "")

    def fit(self, stream: FrameStream, frame_quantile: float, seed: int = 0) -> EdgeBaseline:
        started = time.perf_counter()
        self.baseline = calibrate_edge(stream, self.cfg, self.contexts, self.n_components, frame_quantile)
        self.threshold = self.baseline.distance_threshold()
        self.fit_seconds = time.perf_counter() - started
        return self

    def score(self, stream: FrameStream) -> np.ndarray:
        return self.baseline.embedding.score_batch(stream.embeddings)

    def size_bytes(self) -> int:
        return self.baseline.embedding.size_bytes() + self.baseline.score_sample.nbytes


def calibrate_edge(stream: FrameStream, cfg: AnomalyConfig, contexts: str | None = None,
                   n_components: int | None = None, frame_quantile: float | None = None):
    """Runs the device's calibration over all of a stream, split as the device would."""
    stats, threshold = split_calibration(
        observed_seconds(stream.timestamps, cfg.frame_seconds, cfg.max_gap_seconds))
    emb = cfg.embedding
    cfg = replace(
        cfg,
        embedding=replace(emb, contexts=contexts or emb.contexts,
                          n_components=n_components if n_components is not None else emb.n_components,
                          frame_quantile=frame_quantile or emb.frame_quantile),
        calibration=replace(cfg.calibration, stats_seconds=stats, threshold_seconds=threshold,
                            min_frames=min(cfg.calibration.min_frames, len(stream) // 2),
                            min_threshold_frames=min(cfg.calibration.min_threshold_frames, len(stream) // 8)),
    )
    cal = BaselineCalibrator(cfg)
    for frame in stream.frames():
        cal.add(frame)
    return cal.finalize()


class Euclidean(Scorer):
    """Distance to the mean embedding: the naive reference."""

    name = "euclidean"

    def _fit(self, stream, seed):
        self.mean = stream.embeddings.mean(axis=0)

    def score(self, stream):
        return np.linalg.norm(stream.embeddings - self.mean, axis=1)


class KMeansSelective(Scorer):
    """Selective Mahalanobis with unsupervised contexts: k-means clusters instead of YAMNet
    categories. Would need clustering on the Pi; evaluated to see whether it's worth it."""

    def __init__(self, k: int = 8, dims: int = 32):
        self.k, self.dims, self.name = k, dims, f"kmeans{k}_selective"

    def _fit(self, stream, seed):
        x = stream.embeddings
        kmeans = MiniBatchKMeans(self.k, random_state=seed, n_init=3).fit(_subsample(x, 100_000, seed))
        labels = kmeans.predict(x)
        comps = {"global": _gaussian(x, None)}
        for c in range(self.k):
            members = x[labels == c]
            if len(members) >= 50:
                comps[f"c{c}"] = _gaussian(members, min(self.dims, max(4, len(members) // 20)))
        self.model = EmbeddingModel(comps, "kmeans")

    def score(self, stream):
        return self.model.score_batch(stream.embeddings)

    def size_bytes(self):
        return self.model.size_bytes()


class IsolationForestScorer(Scorer):
    name = "isolation_forest"

    def _fit(self, stream, seed):
        self.model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        self.model.fit(_subsample(stream.embeddings, 50_000, seed))

    def score(self, stream):
        return -self.model.score_samples(stream.embeddings)


class KNNScorer(Scorer):
    """Mean distance to the k nearest calibration frames, in a PCA space (memory-bank method)."""

    def __init__(self, k: int = 5, pca_dims: int = 64, bank: int = 20_000):
        self.k, self.pca_dims, self.bank = k, pca_dims, bank
        self.name = f"knn{k}_pca{pca_dims}"

    def _fit(self, stream, seed):
        x = _subsample(stream.embeddings, self.bank, seed)
        self.pca = PCA(min(self.pca_dims, x.shape[1], len(x)), random_state=seed).fit(x)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.pca.transform(x))

    def score(self, stream):
        distances, _ = self.nn.kneighbors(self.pca.transform(stream.embeddings))
        return distances.mean(axis=1)

    def size_bytes(self):
        return self.bank * self.pca.n_components_ * 4 + self.pca.components_.nbytes


def default_methods(has_scores: bool) -> list[Scorer]:
    methods: list[Scorer] = [EdgeBaseline("global"), EdgeBaseline("global", n_components=64),
                             EdgeBaseline("period")]
    if has_scores:
        methods.insert(0, EdgeBaseline("category"))
    return methods + [KMeansSelective(), IsolationForestScorer(), KNNScorer(), Euclidean()]


def _gaussian(x: np.ndarray, dims: int | None) -> GaussianModel:
    moments = RunningMoments(x.shape[1])
    moments.add_batch(x)
    return GaussianModel.fit(moments, dims)


def _subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[np.sort(np.random.default_rng(seed).choice(len(x), n, replace=False))]
