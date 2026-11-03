"""Reconstruct the workflow in Fazel et al., arXiv:2602.06955.

The paper does not release OA entries or parameter levels. The explicit OA and
levels below are OUR reconstruction, not recovered author configuration.
"""
import json
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)


def orthogonal_array(prime, columns):
    """OA(p², columns, p, 2), generated over GF(p); p=3 or 5 here."""
    if prime not in (3, 5) or not 2 <= columns <= prime+1:
        raise ValueError("Supported prime/column count exceeded")
    return np.array([[a, b]+[(a+k*b) % prime for k in range(1, columns-1)]
                     for a in range(prime) for b in range(prime)], dtype=int)


def scaler_order(row):
    """Interpret each position as a scaler; skip a repeated scaler (identity)."""
    return list(dict.fromkeys(int(v) for v in row))


def make_scalers(order, n_rows, seed):
    factory = {
        0: lambda: MinMaxScaler(),
        1: lambda: StandardScaler(),
        2: lambda: PowerTransformer(method="yeo-johnson"),
        3: lambda: QuantileTransformer(n_quantiles=min(1000, n_rows),
                                      output_distribution="normal", random_state=seed),
        4: lambda: RobustScaler(quantile_range=(25, 75)),
    }
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))]+
                    [(f"scale_{i}_{v}", factory[v]()) for i, v in enumerate(order)])


class OptimizedEBM2026(ClassifierMixin, BaseEstimator):
    """Train-only inner-CV L25 + L9 search, then top-18 selection and refit."""

    def __init__(self, seed=42, n_jobs=2, quick=False, cv_folds=3, progress_path=None):
        self.seed = seed
        self.n_jobs = n_jobs
        self.quick = quick
        self.cv_folds = cv_folds
        self.progress_path = progress_path

    def _model(self, params):
        settings = dict(interactions=20, max_bins=256, learning_rate=0.05, max_rounds=100)
        settings.update(params)
        if self.quick:
            settings["max_rounds"] = 5
            settings["interactions"] = min(2, settings["interactions"])
        return ExplainableBoostingClassifier(**settings, random_state=self.seed,
            n_jobs=self.n_jobs, outer_bags=1 if self.quick else 14)

    def _progress(self, done, total, stage):
        message = f"{done}/{total} optimized_ebm_2026 {stage}"
        print(message, flush=True)
        if self.progress_path:
            with Path(self.progress_path).open("a", encoding="utf-8") as stream:
                stream.write(message+"\n")

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("EBM needs normal/fraud labels")
        if np.bincount(y).min() < self.cv_folds:
            raise ValueError("Too few minority observations for inner CV")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        cv = list(StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.seed).split(X, y))
        oa25 = orthogonal_array(5, 5)[:2 if self.quick else 25]
        oa9 = orthogonal_array(3, 4)[:2 if self.quick else 9]
        self.history_ = []
        done, total = 0, (len(oa25)+len(oa9))*self.cv_folds+2

        def cross_validate(order, params, stage, trial):
            nonlocal done
            scores = []
            for fold, (train, held) in enumerate(cv):
                transform = make_scalers(order, len(train), self.seed)
                tx = transform.fit_transform(X[train])
                vx = transform.transform(X[held])
                model = self._model(params).fit(tx, y[train])
                scores.append(float(roc_auc_score(y[held], model.predict_proba(vx)[:, 1])))
                done += 1
                self._progress(done, total, f"{stage} trial={trial} fold={fold}")
            self.history_.append(dict(stage=stage, trial=trial, scaler_order=list(order),
                parameters=params, fold_roc_auc=scores, mean_roc_auc=float(np.mean(scores))))
            return float(np.mean(scores))

        scaler_trials = [(cross_validate(scaler_order(row), {}, "L25", i), scaler_order(row))
                         for i, row in enumerate(oa25)]
        self.scaler_order_ = max(scaler_trials, key=lambda item: item[0])[1]
        # Levels are disclosed reconstruction choices because the paper only gives
        # the winning configuration, not the original three levels per factor.
        self.levels_ = dict(interactions=[0, 10, 20], max_bins=[64, 128, 256],
                            learning_rate=[0.01, 0.05, 0.1], max_rounds=[100, 50, 200])
        parameters = [{key: values[int(level)] for (key, values), level in zip(self.levels_.items(), row)}
                      for row in oa9]
        trials = [(cross_validate(self.scaler_order_, params, "L9", i), params)
                  for i, params in enumerate(parameters)]
        self.best_parameters_ = max(trials, key=lambda item: item[0])[1]
        self.scalers_ = make_scalers(self.scaler_order_, len(y), self.seed)
        transformed = self.scalers_.fit_transform(X)
        initial = self._model(self.best_parameters_).fit(transformed, y)
        # Aggregate each pair's importance equally onto its participating features.
        importance = np.zeros(X.shape[1])
        for term, weight in zip(initial.term_features_, initial.term_importances()):
            for feature in term:
                importance[feature] += float(weight)/len(term)
        self.feature_importances_ = importance
        self.selected_features_ = np.argsort(-importance, kind="stable")[:min(18, X.shape[1])]
        done += 1
        self._progress(done, total, "train_only_top18_ranking")
        self.model_ = self._model(self.best_parameters_).fit(transformed[:, self.selected_features_], y)
        done += 1
        self._progress(done, total, "final_refit")
        return self

    def predict_proba(self, X):
        transformed = self.scalers_.transform(np.asarray(X, dtype=float))
        return self.model_.predict_proba(transformed[:, self.selected_features_])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
