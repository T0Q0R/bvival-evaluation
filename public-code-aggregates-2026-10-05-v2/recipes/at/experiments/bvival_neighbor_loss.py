"""Cached OOF loss kNN adaptation, not a full TAFA/ACO reproduction.

Only training loss caches enter fit. predict accepts strictly pre-action contexts;
it has no target, acquisition-value, or evaluation-outcome argument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_chunked
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CachedNeighborLoss:
    def __init__(self, categorical, numeric, *, max_categories=64, working_memory=64):
        self.categorical = list(categorical)
        self.numeric = list(numeric)
        self.fields = self.categorical + self.numeric
        if (not self.categorical or not self.numeric or len(set(self.fields)) != len(self.fields)
                or isinstance(max_categories, bool) or not isinstance(max_categories, int) or max_categories < 2
                or not np.isfinite(working_memory) or working_memory <= 0):
            raise ValueError('Distinct fields and positive preprocessing parameters required')
        self.max_categories, self.working_memory = max_categories, working_memory

    def _frame(self, contexts):
        if not contexts or any(set(c) != set(self.fields) for c in contexts):
            raise ValueError('Exact nonempty pre-action context allowlist required')
        rows = []
        for context in contexts:
            row = {}
            for field in self.categorical:
                value = context[field]
                if not isinstance(value, str):
                    raise ValueError('Categorical values must be normalized strings')
                row[field] = value if value else '__MISSING__'
            for field in self.numeric:
                value = context[field]
                if value == '':
                    row[field] = np.nan
                else:
                    if isinstance(value, bool):
                        raise ValueError('Boolean numeric context is not allowed')
                    number = float(value)
                    if not np.isfinite(number):
                        raise ValueError('Numeric context must be finite or empty')
                    row[field] = number
            rows.append(row)
        return pd.DataFrame(rows, columns=self.fields)

    def fit(self, keys, contexts, before_loss, after_loss):
        n = len(keys)
        if (not n or len(set(keys)) != n or len(contexts) != n
                or any(not isinstance(k, str) or not k for k in keys)):
            raise ValueError('Unique aligned training keys required')
        before, after = np.asarray(before_loss, float), np.asarray(after_loss, float)
        if (before.shape != (n,) or after.shape != (n, 3)
                or not np.isfinite(before).all() or not np.isfinite(after).all()
                or (before < 0).any() or (after < 0).any()):
            raise ValueError('Aligned finite nonnegative OOF loss cache required')
        order = sorted(range(n), key=lambda i: keys[i])
        self.keys_ = [keys[i] for i in order]
        frame = self._frame([contexts[i] for i in order])
        self.preprocessor_ = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=self.max_categories,
                                  sparse_output=False), self.categorical),
            ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True,
                                                      keep_empty_features=True)),
                              ('scale', StandardScaler())]), self.numeric),
        ], sparse_threshold=0)
        self.train_ = np.asarray(self.preprocessor_.fit_transform(frame), float)
        if not np.isfinite(self.train_).all():
            raise ValueError('Nonfinite training representation')
        self.cache_ = np.column_stack([before[order], after[order]])
        return self

    def predict(self, contexts, neighbor_counts):
        if not hasattr(self, 'train_'):
            raise ValueError('Fit training-only representation/cache first')
        counts = list(neighbor_counts)
        if (not counts or len(set(counts)) != len(counts) or any(isinstance(k, bool)
                or not isinstance(k, int) or not 1 <= k <= len(self.train_) for k in counts)):
            raise ValueError('Distinct positive feasible integer neighbor counts required')
        query = np.asarray(self.preprocessor_.transform(self._frame(contexts)), float)
        if not np.isfinite(query).all():
            raise ValueError('Nonfinite query representation')
        top = max(counts)
        scores = {k: np.empty((len(query), 3)) for k in counts}
        start = 0
        for distances in pairwise_distances_chunked(query, self.train_, metric='euclidean',
                n_jobs=1, working_memory=self.working_memory):
            for local, distance in enumerate(distances):
                # Include every cutoff tie, then use canonical training order.
                chosen = np.argpartition(distance, top - 1)[:top]
                eligible = np.flatnonzero(distance <= distance[chosen].max())
                neighbors = eligible[np.lexsort((eligible, distance[eligible]))][:top]
                cumulative = np.cumsum(self.cache_[neighbors], axis=0)
                for k in counts:
                    means = cumulative[k - 1] / k
                    scores[k][start + local] = means[0] - means[1:]
            start += len(distances)
        return scores
