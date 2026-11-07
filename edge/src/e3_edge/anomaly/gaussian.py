"""The acoustic baseline: mean + covariance of YAMNet embeddings, and Mahalanobis scoring.

Proposal 1.7.3.2: a new embedding is scored by its Mahalanobis distance from the learned mean,
using the covariance of household embeddings. Three practical details:

* Streaming: frames are never stored. Mean and covariance are accumulated in batches with the
  pairwise update of Chan et al., in float64 for accuracy over days of audio.
* Shrinkage: a 1024x1024 covariance is noisy and often singular (some YAMNet dimensions are
  almost always zero). It is shrunk with OAS (Chen et al., 2010), which needs only the
  covariance and the frame count, so it also works on streamed statistics. Checked against
  scikit-learn in the tests.
* Optional PCA: with n_components=k, the distance uses the top k directions exactly and the
  rest through their average variance (probabilistic PCA). This keeps sensitivity to sounds
  that leave the household subspace while making the stored model 1024/k times smaller.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class RunningMoments:
    """Streaming mean and covariance of D-dimensional vectors."""

    def __init__(self, dim: int, batch_size: int = 256):
        self.dim = dim
        self.batch_size = batch_size
        self._n = 0
        self._mean = np.zeros(dim)
        self._m2 = np.zeros((dim, dim))
        self._buffer: list[np.ndarray] = []

    @property
    def n(self) -> int:
        return self._n + len(self._buffer)

    def add(self, x: np.ndarray) -> None:
        self._buffer.append(np.asarray(x, dtype=np.float64))
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def add_batch(self, xs: np.ndarray) -> None:
        self.flush()
        xs = np.asarray(xs, dtype=np.float64)
        if xs.ndim != 2 or xs.shape[1] != self.dim:
            raise ValueError(f"expected [N, {self.dim}] vectors")
        if len(xs):
            self._merge(len(xs), xs.mean(axis=0), _scatter(xs))

    def flush(self) -> None:
        if self._buffer:
            xs = np.stack(self._buffer)
            self._buffer.clear()
            self._merge(len(xs), xs.mean(axis=0), _scatter(xs))

    def _merge(self, n_b: int, mean_b: np.ndarray, m2_b: np.ndarray) -> None:
        n_a, n = self._n, self._n + n_b
        delta = mean_b - self._mean
        self._mean += delta * (n_b / n)
        self._m2 += m2_b + np.outer(delta, delta) * (n_a * n_b / n)
        self._n = n

    @property
    def mean(self) -> np.ndarray:
        self.flush()
        return self._mean.copy()

    def covariance(self) -> np.ndarray:
        """Maximum-likelihood (1/n) covariance, the form OAS expects."""
        self.flush()
        if self._n < 2:
            raise ValueError("need at least 2 vectors for a covariance")
        return self._m2 / self._n

    def state(self) -> dict[str, np.ndarray]:
        self.flush()
        return {"n": np.array(self._n), "mean": self._mean, "m2": self._m2}

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray], batch_size: int = 256) -> RunningMoments:
        mean = np.asarray(state["mean"], dtype=np.float64)
        moments = cls(len(mean), batch_size)
        moments._n = int(state["n"])
        moments._mean = mean.copy()
        moments._m2 = np.asarray(state["m2"], dtype=np.float64).copy()
        return moments


def _scatter(xs: np.ndarray) -> np.ndarray:
    centred = xs - xs.mean(axis=0)
    return centred.T @ centred


def oas_shrink(cov: np.ndarray, n_samples: int) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage towards a scaled identity (same formula as scikit-learn)."""
    p = cov.shape[0]
    alpha = float(np.mean(cov**2))
    mu = float(np.trace(cov)) / p
    num = alpha + mu**2
    den = (n_samples + 1) * (alpha - mu**2 / p)
    shrinkage = 1.0 if den <= 0 else min(num / den, 1.0)
    shrunk = (1.0 - shrinkage) * cov
    shrunk.flat[:: p + 1] += shrinkage * mu
    return shrunk, shrinkage


class GaussianModel:
    """One Gaussian over embeddings, scored by Mahalanobis distance."""

    def __init__(self, mean: np.ndarray, eigvecs: np.ndarray, eigvals: np.ndarray,
                 residual_var: float = 0.0, n_frames: int = 0, shrinkage: float = 0.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.eigvecs = np.asarray(eigvecs, dtype=np.float32)  # [D, k], columns = directions
        self.eigvals = np.asarray(eigvals, dtype=np.float64)  # [k], descending
        self.residual_var = float(residual_var)
        self.n_frames = int(n_frames)
        self.shrinkage = float(shrinkage)
        if self.eigvecs.shape != (len(self.mean), len(self.eigvals)):
            raise ValueError("eigvecs must be [D, k] to match mean and eigvals")
        if np.any(self.eigvals <= 0) or (self.reduced and self.residual_var <= 0):
            raise ValueError("variances must be positive")
        self._inv = (1.0 / self.eigvals).astype(np.float32)

    @property
    def dim(self) -> int:
        return len(self.mean)

    @property
    def reduced(self) -> bool:
        return len(self.eigvals) < self.dim

    @classmethod
    def fit(cls, moments: RunningMoments, n_components: int | None = None,
            ridge: float = 1e-6) -> GaussianModel:
        return cls.from_covariance(moments.mean, moments.covariance(), moments.n, n_components, ridge)

    @classmethod
    def from_covariance(cls, mean: np.ndarray, cov: np.ndarray, n_frames: int,
                        n_components: int | None = None, ridge: float = 1e-6,
                        shrink: bool = True) -> GaussianModel:
        dim = len(mean)
        shrinkage = 0.0
        if shrink:
            cov, shrinkage = oas_shrink(cov, n_frames)
        else:
            cov = cov.copy()
        scale = float(np.trace(cov)) / dim
        cov.flat[:: dim + 1] += ridge * (scale if scale > 0 else 1.0)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 1e-12 * max(scale, 1e-12))
        eigvecs = eigvecs[:, order]
        k = dim if n_components is None else max(1, min(int(n_components), dim))
        residual = float(eigvals[k:].mean()) if k < dim else 0.0
        return cls(mean, eigvecs[:, :k], eigvals[:k], residual, n_frames, shrinkage)

    def covariance(self) -> np.ndarray:
        """The (regularised) covariance this model scores with."""
        v = self.eigvecs.astype(np.float64)
        cov = (v * self.eigvals) @ v.T
        if self.reduced:
            cov += self.residual_var * (np.eye(self.dim) - v @ v.T)
        return cov

    def distance_squared(self, x: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of one [D] vector or a batch [N, D]."""
        d = np.asarray(x, dtype=np.float32) - self.mean
        p = d @ self.eigvecs
        out = (p * p) @ self._inv
        if self.reduced:
            residual = np.einsum("...i,...i->...", d, d) - np.einsum("...i,...i->...", p, p)
            out = out + np.maximum(residual, 0.0) / self.residual_var
        return out.astype(np.float64)

    def distance(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(self.distance_squared(x))

    def blend(self, other_mean: np.ndarray, other_cov: np.ndarray, weight: float,
              n_frames: int, n_components: int | None = None, ridge: float = 1e-6) -> GaussianModel:
        """Mix this model with new statistics, `weight` being the share of the new ones."""
        w = float(np.clip(weight, 0.0, 1.0))
        m_old = self.mean.astype(np.float64)
        m_new = np.asarray(other_mean, dtype=np.float64)
        delta = m_new - m_old
        cov = (1 - w) * self.covariance() + w * np.asarray(other_cov) + w * (1 - w) * np.outer(delta, delta)
        k = n_components if n_components is not None else (len(self.eigvals) if self.reduced else None)
        return GaussianModel.from_covariance((1 - w) * m_old + w * m_new, cov,
                                             self.n_frames + n_frames, k, ridge, shrink=False)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean, "eigvecs": self.eigvecs, "eigvals": self.eigvals,
                "residual_var": np.array(self.residual_var), "n_frames": np.array(self.n_frames),
                "shrinkage": np.array(self.shrinkage)}

    @classmethod
    def from_arrays(cls, a: Mapping[str, np.ndarray]) -> GaussianModel:
        return cls(a["mean"], a["eigvecs"], a["eigvals"], float(a["residual_var"]),
                   int(a["n_frames"]), float(a["shrinkage"]))


class EmbeddingModel:
    """The embedding baseline: one Gaussian, or several with the smallest distance winning."""

    def __init__(self, components: Mapping[str, GaussianModel], mode: str = "global"):
        if not components or (mode == "global" and "global" not in components):
            raise ValueError("embedding model needs a 'global' component")
        dims = {c.dim for c in components.values()}
        if len(dims) != 1:
            raise ValueError("all components must have the same dimension")
        self.components = dict(components)
        self.mode = mode
        self.dim = dims.pop()

    def score(self, x: np.ndarray) -> tuple[float, str]:
        """(Mahalanobis distance, name of the closest component) for one embedding."""
        if self.mode == "global":
            return float(self.components["global"].distance(x)), "global"
        best_name, best = "", np.inf
        for name, comp in self.components.items():
            d2 = float(comp.distance_squared(x))
            if d2 < best:
                best_name, best = name, d2
        return float(np.sqrt(best)), best_name

    def score_batch(self, xs: np.ndarray) -> np.ndarray:
        if self.mode == "global":
            return self.components["global"].distance(xs)
        return np.sqrt(np.min([c.distance_squared(xs) for c in self.components.values()], axis=0))

    def size_bytes(self) -> int:
        return sum(a.nbytes for c in self.components.values() for a in c.arrays().values())
