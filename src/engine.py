"""
FactCheckingEngine
===================
Given a knowledge graph of (subject, predicate, object) facts, some of
them labelled true/false, this engine learns to score the plausibility
("veracity") of arbitrary new (subject, predicate, object) facts, on a
continuous 0..1 scale.

Pipeline per fact:
  1. Structural pool features   (Pool)              -- entity/predicate popularity
  2. Label-derived KG features  (LabelContext)       -- known true/false evidence
  3. KG-embedding plausibility  (TransEClassifier)   -- learned latent geometry
  4. Predicate one-hot
  -> concatenated feature vector -> meta-classifier (Logistic Regression /
     Histogram Gradient Boosting) -> calibrated probability = veracity score
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from features import Pool, LabelContext
from kge import TransEClassifier

PREDICATES = ['deathPlace', 'birthPlace', 'award', 'starring', 'team',
              'author', 'foundationPlace', 'spouse', 'subsidiary']
# (fixed, known vocabulary of the 9 relations in this benchmark)


def _predicate_local(p):
    return p.rsplit('/', 1)[-1]


def _onehot_predicate(df):
    local = df['predicate'].map(_predicate_local)
    out = pd.DataFrame(index=df.index)
    for pr in PREDICATES:
        out[f'pred_{pr}'] = (local == pr).astype(float)
    return out


def train_kge(fold_df, pool, **kge_kwargs):
    s_idx = fold_df['subject'].map(pool.entity2idx).values
    p_idx = fold_df['predicate'].map(pool.pred2idx).values
    o_idx = fold_df['object'].map(pool.entity2idx).values
    y = fold_df['truth_value'].astype(float).values
    model = TransEClassifier(**kge_kwargs)
    model.fit(len(pool.entity2idx), len(pool.pred2idx), s_idx, p_idx, o_idx, y)
    return model


def kge_features(df, pool, kge_model):
    s_idx = df['subject'].map(pool.entity2idx).values
    p_idx = df['predicate'].map(pool.pred2idx).values
    o_idx = df['object'].map(pool.entity2idx).values
    score = kge_model.score(s_idx, p_idx, o_idx)
    proba = kge_model.predict_proba(s_idx, p_idx, o_idx)
    return pd.DataFrame({'kge_score': score, 'kge_proba': proba}, index=df.index)


def build_features(df, pool, label_ctx, kge_model, exclude_self=False, use_kge=True):
    parts = [
        pool.featurize(df),
        label_ctx.featurize(df, exclude_self=exclude_self),
        _onehot_predicate(df),
    ]
    if use_kge and kge_model is not None:
        parts.insert(2, kge_features(df, pool, kge_model))
    return pd.concat(parts, axis=1)


class FactCheckingEngine:
    """
    use_kge=False is the empirically-chosen default: a 5-fold CV ablation
    on the training data (see ablation.py) showed the TransE-style embedding
    score actually *hurts* the combined model slightly on this dataset
    (~1200 labelled triples over ~2000 entities is too sparse for 32-dim
    embeddings to be trained reliably -- most entities appear only once or
    twice). The embedding module is kept and can be switched on with
    use_kge=True; it may become useful on larger/denser knowledge graphs.
    """
    def __init__(self, model_type='logreg', use_kge=False, kge_dim=32,
                 kge_epochs=500, kge_lr=0.2, kge_l2=1e-3, seed=0):
        self.model_type = model_type
        self.use_kge = use_kge
        self.kge_kwargs = dict(dim=kge_dim, epochs=kge_epochs, lr=kge_lr,
                                l2=kge_l2, seed=seed)
        self.seed = seed

    def _make_classifier(self):
        if self.model_type == 'logreg':
            base = LogisticRegression(max_iter=2000, C=1.0)
        elif self.model_type == 'gboost':
            base = HistGradientBoostingClassifier(max_depth=4, max_iter=150,
                                                   learning_rate=0.08,
                                                   random_state=self.seed)
        else:
            raise ValueError(self.model_type)
        return base

    def fit(self, train_df, pool):
        """train_df must have subject, predicate, object, truth_value (0/1)."""
        self.pool = pool
        self.label_ctx = LabelContext(train_df)
        self.kge_model = train_kge(train_df, pool, **self.kge_kwargs) if self.use_kge else None

        X = build_features(train_df, pool, self.label_ctx, self.kge_model,
                            exclude_self=True, use_kge=self.use_kge)
        y = train_df['truth_value'].astype(float).values
        self.feature_names_ = X.columns.tolist()

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X.values)

        base = self._make_classifier()
        # calibrate probabilities via 5-fold internal CV so output behaves
        # like a genuine probability, not just a raw classifier score
        self.clf = CalibratedClassifierCV(base, method='isotonic', cv=5)
        self.clf.fit(Xs, y)
        return self

    def predict_veracity(self, df):
        """df: DataFrame with subject, predicate, object (URIs or local names
        already mapped to the same space used in fit). Returns np.array of
        veracity scores in [0,1]."""
        X = build_features(df, self.pool, self.label_ctx, self.kge_model,
                            exclude_self=False, use_kge=self.use_kge)
        X = X[self.feature_names_]
        Xs = self.scaler.transform(X.values)
        return self.clf.predict_proba(Xs)[:, 1]
