import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, brier_score_loss, log_loss)

from parse_nt import parse_statements
from features import Pool
from engine import FactCheckingEngine

# Resolve the data files relative to this repo, regardless of the current
# working directory the scripts are run from (repo root, src/, etc.), with a
# couple of common fallbacks.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [
    os.path.join(_THIS_DIR, '..', 'data'),   # repo_root/data (expected layout)
    os.path.join(_THIS_DIR, 'data'),
    _THIS_DIR,
    '.',
    'data',
]


def _find_file(fname):
    for d in _CANDIDATE_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"Could not find {fname!r}. Place KG-2022-train_nt.txt and "
        f"KG-2022-test_nt.txt in the repo's data/ folder."
    )


TRAIN_PATH = _find_file('KG-2022-train_nt.txt')
TEST_PATH = _find_file('KG-2022-test_nt.txt')


def load_pool():
    train = parse_statements(TRAIN_PATH)
    test = parse_statements(TEST_PATH)
    all_df = pd.concat([train, test], ignore_index=True)
    return train, test, Pool(all_df)


def metrics_report(y_true, p_pred, name=''):
    y_hat = (p_pred >= 0.5).astype(int)
    return {
        'model': name,
        'accuracy': accuracy_score(y_true, y_hat),
        'precision': precision_score(y_true, y_hat, zero_division=0),
        'recall': recall_score(y_true, y_hat, zero_division=0),
        'f1': f1_score(y_true, y_hat, zero_division=0),
        'roc_auc': roc_auc_score(y_true, p_pred),
        'brier': brier_score_loss(y_true, p_pred),
        'log_loss': log_loss(y_true, p_pred, labels=[0, 1]),
    }


def cross_validate(train, pool, model_type='logreg', n_splits=5, seed=0):
    strat_key = train['predicate'] + '__' + train['truth_value'].astype(int).astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(train))
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, strat_key)):
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)

        eng = FactCheckingEngine(model_type=model_type, seed=seed)
        eng.fit(tr_df, pool)
        p_val = eng.predict_veracity(va_df)
        oof_pred[va_idx] = p_val

        m = metrics_report(va_df['truth_value'].astype(int).values, p_val,
                            name=f'{model_type} fold{fold}')
        fold_metrics.append(m)

    overall = metrics_report(train['truth_value'].astype(int).values, oof_pred,
                              name=f'{model_type} (5-fold OOF overall)')
    return oof_pred, pd.DataFrame(fold_metrics), overall


def baseline_majority(train):
    p = np.full(len(train), train['truth_value'].mean())
    return metrics_report(train['truth_value'].astype(int).values, p, name='majority/prior baseline')


def baseline_predicate_prior(train, n_splits=5, seed=0):
    # out-of-fold predicate-prior baseline (still avoids leakage)
    strat_key = train['predicate'] + '__' + train['truth_value'].astype(int).astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(train, strat_key):
        prior = train.iloc[tr_idx].groupby('predicate')['truth_value'].mean()
        oof[va_idx] = train.iloc[va_idx]['predicate'].map(prior).fillna(train.iloc[tr_idx]['truth_value'].mean()).values
    return oof, metrics_report(train['truth_value'].astype(int).values, oof, name='predicate-prior baseline (OOF)')


if __name__ == '__main__':
    train, test, pool = load_pool()
    print(f"train={len(train)}  test={len(test)}  entities={len(pool.entity2idx)}  predicates={len(pool.pred2idx)}")

    print("\n--- Baselines ---")
    print(pd.Series(baseline_majority(train)))
    _, bp = baseline_predicate_prior(train)
    print(pd.Series(bp))

    print("\n--- 5-fold CV: Logistic Regression meta-model ---")
    oof_lr, folds_lr, overall_lr = cross_validate(train, pool, model_type='logreg')
    print(folds_lr)
    print(pd.Series(overall_lr))

    print("\n--- 5-fold CV: Gradient Boosting meta-model ---")
    oof_gb, folds_gb, overall_gb = cross_validate(train, pool, model_type='gboost')
    print(folds_gb)
    print(pd.Series(overall_gb))

    print("\n--- Per-relation breakdown (logreg OOF) ---")
    tmp = train.copy()
    tmp['pred_score'] = oof_lr
    tmp['pred_label'] = (tmp['pred_score'] >= 0.5).astype(int)
    per_rel = tmp.groupby('predicate_name').apply(
        lambda g: pd.Series({
            'n': len(g),
            'accuracy': accuracy_score(g['truth_value'].astype(int), g['pred_label']),
            'roc_auc': roc_auc_score(g['truth_value'].astype(int), g['pred_score']) if g['truth_value'].nunique() > 1 else np.nan,
        }), include_groups=False)
    print(per_rel.sort_values('n', ascending=False))
