import os
import pandas as pd
from evaluate import load_pool
from engine import FactCheckingEngine

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_THIS_DIR, '..', 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(_OUT_DIR, 'test_predictions.csv')


def main():
    train, test, pool = load_pool()

    eng = FactCheckingEngine(model_type='logreg', use_kge=False, seed=0)
    eng.fit(train, pool)

    veracity = eng.predict_veracity(test)
    out = test[['stmt_id', 'subject_name', 'predicate_name', 'object_name']].copy()
    out['veracity_score'] = veracity
    out['predicted_label'] = (out['veracity_score'] >= 0.5).map({True: 'true', False: 'false'})
    out = out.sort_values('veracity_score', ascending=False).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"wrote {len(out)} predictions to {OUT_PATH}")
    print(out.head(10))
    print('...')
    print(out.tail(10))
    print('\nPredicted-true rate on test:', (out['predicted_label'] == 'true').mean())
    print('Veracity score distribution:\n', out['veracity_score'].describe())
    return eng, out


if __name__ == '__main__':
    main()
