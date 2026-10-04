"""Shared observed-space neighbors with action-eligible OOF gain caches.

Local one-step adaptation, not TAFA/ACO reproduction. Unknown action outcomes
are omitted, never zero-filled. An empty neighbor/action cache uses that
action's development-only global mean; the fallback count is disclosed.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances_chunked

from bvival_neighbor_loss import CachedNeighborLoss


class AvailableNeighborGain(CachedNeighborLoss):
    def fit_gains(self, keys, contexts, gains):
        values = np.asarray(gains, float)
        if (values.ndim != 2 or values.shape[0] != len(keys) or values.shape[1] < 2
                or np.isinf(values).any() or np.isnan(values).all(axis=1).any()
                or np.isnan(values).all(axis=0).any()):
            raise ValueError('Finite or unavailable aligned action gains required')
        # Reuse the checked, train-only representation, not its three-action cache.
        super().fit(keys, contexts, np.zeros(len(keys)), np.zeros((len(keys), 3)))
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        self.gains_ = values[order].copy()
        self.global_ = np.nanmean(self.gains_, axis=0)
        return self

    def predict(self, contexts, neighbor_counts):
        if not hasattr(self, 'gains_'):
            raise ValueError('Fit eligible development OOF gains first')
        counts = list(neighbor_counts)
        if (not counts or len(set(counts)) != len(counts) or any(isinstance(k, bool)
                or not isinstance(k, int) or not 1 <= k <= len(self.train_) for k in counts)):
            raise ValueError('Distinct positive feasible integer neighbor counts required')
        query = np.asarray(self.preprocessor_.transform(self._frame(contexts)), float)
        if not np.isfinite(query).all():
            raise ValueError('Nonfinite query representation')
        scores = {k: np.empty((len(query), self.gains_.shape[1])) for k in counts}
        fallbacks = {k: np.zeros(self.gains_.shape[1], dtype=int) for k in counts}
        start, top = 0, max(counts)
        for distances in pairwise_distances_chunked(query, self.train_, metric='euclidean',
                n_jobs=1, working_memory=self.working_memory):
            for local, distance in enumerate(distances):
                chosen = np.argpartition(distance, top - 1)[:top]
                eligible = np.flatnonzero(distance <= distance[chosen].max())
                neighbors = eligible[np.lexsort((eligible, distance[eligible]))][:top]
                values = self.gains_[neighbors]
                cumulative = np.cumsum(np.nan_to_num(values, nan=0.0), axis=0)
                observed = np.cumsum(~np.isnan(values), axis=0)
                for k in counts:
                    n = observed[k-1]
                    out = self.global_.copy()
                    np.divide(cumulative[k-1], n, out=out, where=n > 0)
                    scores[k][start+local] = out
                    fallbacks[k] += (n == 0)
            start += len(distances)
        self.last_fallback_counts_ = fallbacks
        return scores
