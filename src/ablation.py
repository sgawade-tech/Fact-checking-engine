import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from evaluate import load_pool, metrics_report
from features import Pool, LabelContext
from engine import train_kge, kge_features, _onehot_predicate

BLOCKS = {
    'predicate_onehot_only': lambda pool, lc, kge, df, excl: [_onehot_predicate(df)],
    'pool_only': lambda pool, lc, kge, df, excl: [pool.featurize(df), _onehot_predicate(df)],
    'label_only': lambda pool, lc, kge, df, excl: [lc.featurize(df, exclude_self=excl), _onehot_predicate(df)],
    'pool_plus_label (no KGE)': lambda pool, lc, kge, df, excl: [pool.featurize(df), lc.featurize(df, exclude_self=excl), _onehot_predicate(df)],
    'kge_only': lambda pool, lc, kge, df, excl: [kge_features(df, pool, kge), _onehot_predicate(df)],
    'full (pool+label+kge)': lambda pool, lc, kge, df, excl: [pool.featurize(df), lc.featurize(df, exclude_self=excl), kge_features(df, pool, kge), _onehot_predicate(df)],
}


def run_block(block_fn, train, pool, n_splits=5, seed=0):
    strat_key = train['predicate'] + '__' + train['truth_value'].astype(int).astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(train, strat_key):
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)
        lc = LabelContext(tr_df)
        kge = train_kge(tr_df, pool, dim=32, epochs=500, lr=0.2, l2=1e-3, seed=seed)

        Xtr = pd.concat(block_fn(pool, lc, kge, tr_df, True), axis=1)
        Xva = pd.concat(block_fn(pool, lc, kge, va_df, False), axis=1)
        Xva = Xva[Xtr.columns]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr.values)
        Xva_s = scaler.transform(Xva.values)

        base = LogisticRegression(max_iter=2000)
        clf = CalibratedClassifierCV(base, method='isotonic', cv=5)
        clf.fit(Xtr_s, tr_df['truth_value'].astype(float).values)
        oof[va_idx] = clf.predict_proba(Xva_s)[:, 1]
    return oof, metrics_report(train['truth_value'].astype(int).values, oof)


if __name__ == '__main__':
    import os
    train, test, pool = load_pool()
    results = []
    for name, fn in BLOCKS.items():
        _, m = run_block(fn, train, pool)
        m['model'] = name
        results.append(m)
        print(f"{name:28s}  acc={m['accuracy']:.3f}  auc={m['roc_auc']:.3f}  f1={m['f1']:.3f}  brier={m['brier']:.3f}")
    _out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
    os.makedirs(_out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(_out_dir, 'ablation_results.csv'), index=False)
