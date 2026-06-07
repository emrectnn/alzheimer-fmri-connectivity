"""Network-Based Statistic (NBS) edge selection for group contrasts."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


def vec_to_mat(vec: np.ndarray, n_roi: int) -> np.ndarray:
    iu = np.triu_indices(n_roi, k=1)
    M = np.zeros((n_roi, n_roi))
    M[iu] = vec
    M = M + M.T
    return M


def mat_to_vec(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def compute_edge_tstats(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X0 = X[y == 0]
    X1 = X[y == 1]
    t, _ = stats.ttest_ind(X1, X0, axis=0, equal_var=False)
    return np.nan_to_num(t, nan=0.0)


def connected_components_from_edges(edge_mask: np.ndarray, n_roi: int) -> list[set]:
    iu = np.triu_indices(n_roi, k=1)
    edge_list = [(iu[0][i], iu[1][i]) for i in range(len(edge_mask)) if edge_mask[i]]
    parent = list(range(n_roi))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b) in edge_list:
        union(a, b)

    roots = [find(i) for i in range(n_roi)]
    comp_edges: dict[int, set] = {}
    for idx in range(len(edge_mask)):
        if not edge_mask[idx]:
            continue
        a, b = iu[0][idx], iu[1][idx]
        r = find(a)
        comp_edges.setdefault(r, set()).add(idx)
    return list(comp_edges.values())


class NBSEdgeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_roi: int = 166, thresh: float = 3.1,
                 n_perm: int = 500, alpha: float = 0.05,
                 random_state: int = 42, fallback_topk: int = 30):
        self.n_roi = n_roi
        self.thresh = thresh
        self.n_perm = n_perm
        self.alpha = alpha
        self.random_state = random_state
        self.fallback_topk = fallback_topk

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y).ravel()
        if len(np.unique(y)) < 2:
            self.mask_ = np.ones(X.shape[1], dtype=bool)
            return self

        t_obs = compute_edge_tstats(X, y)
        obs_edge_mask = np.abs(t_obs) > self.thresh

        obs_components = connected_components_from_edges(obs_edge_mask, self.n_roi)
        obs_sizes = [len(c) for c in obs_components]

        if not obs_components:
            top_idx = np.argsort(np.abs(t_obs))[-self.fallback_topk:]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_idx] = True
            self.mask_ = mask
            return self

        rng = np.random.default_rng(self.random_state)
        null_max_sizes = np.zeros(self.n_perm)
        for p in range(self.n_perm):
            y_perm = rng.permutation(y)
            t_perm = compute_edge_tstats(X, y_perm)
            perm_mask = np.abs(t_perm) > self.thresh
            perm_comps = connected_components_from_edges(perm_mask, self.n_roi)
            null_max_sizes[p] = max((len(c) for c in perm_comps), default=0)

        mask = np.zeros(X.shape[1], dtype=bool)
        for comp in obs_components:
            size = len(comp)
            p_val = (np.sum(null_max_sizes >= size) + 1) / (self.n_perm + 1)
            if p_val < self.alpha:
                for idx in comp:
                    mask[idx] = True

        if mask.sum() == 0:
            top_idx = np.argsort(np.abs(t_obs))[-self.fallback_topk:]
            mask[top_idx] = True

        self.mask_ = mask
        self.n_selected_ = int(mask.sum())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'mask_'):
            raise RuntimeError("NBSEdgeSelector: fit cagir once")
        return X[:, self.mask_]

    def get_support(self) -> np.ndarray:
        return self.mask_
