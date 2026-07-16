"""
Feature engineering for KG fact-checking.

Two kinds of features, kept carefully separate to avoid leakage:

1. STRUCTURAL / POOL features -- computed from the mere *existence* of
   triples across the whole known universe (train + test), never using
   truth labels. Safe to compute using the full pool at any time
   (this is the standard "transductive" setting used in KG embedding
   research: the graph structure of the whole dataset is known upfront,
   only the labels of the evaluation triples are hidden).

2. LABEL-DERIVED features -- computed only from a set of *known-labelled*
   triples (a training fold). These use Counters (not sets) so that when
   scoring rows that are themselves part of the label source, a row's own
   contribution can be subtracted out (proper leave-one-out), preventing
   a row's label from leaking into its own features.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict


class Pool:
    """Structural statistics from the union of all (train+test) triples,
    ignoring truth labels entirely."""
    def __init__(self, all_df):
        self.subject_freq = all_df['subject'].value_counts().to_dict()
        self.object_freq = all_df['object'].value_counts().to_dict()
        self.pred_obj_freq = all_df.groupby(['predicate', 'object']).size().to_dict()
        self.subj_pred_freq = all_df.groupby(['subject', 'predicate']).size().to_dict()
        entities = pd.unique(pd.concat([all_df['subject'], all_df['object']]))
        self.entity2idx = {e: i for i, e in enumerate(entities)}
        predicates = pd.unique(all_df['predicate'])
        self.pred2idx = {p: i for i, p in enumerate(predicates)}

    def featurize(self, df):
        out = pd.DataFrame(index=df.index)
        out['subject_freq'] = df['subject'].map(self.subject_freq).fillna(0)
        out['object_freq'] = df['object'].map(self.object_freq).fillna(0)
        out['pred_obj_freq'] = [self.pred_obj_freq.get((p, o), 0)
                                 for p, o in zip(df['predicate'], df['object'])]
        out['subj_pred_freq'] = [self.subj_pred_freq.get((s, p), 0)
                                  for s, p in zip(df['subject'], df['predicate'])]
        out['log_subject_freq'] = np.log1p(out['subject_freq'])
        out['log_object_freq'] = np.log1p(out['object_freq'])
        out['log_pred_obj_freq'] = np.log1p(out['pred_obj_freq'])
        return out


class LabelContext:
    """Statistics derived ONLY from a set of known-labelled triples
    (e.g. a training fold). Must be refit per-fold during CV.
    Uses Counters (not sets) so multiplicities are tracked and a row's
    own contribution can be subtracted when exclude_self=True."""

    def __init__(self, labelled_df):
        self.df = labelled_df
        true_df = labelled_df[labelled_df.truth_value == 1]
        false_df = labelled_df[labelled_df.truth_value == 0]

        self.exact_true_ctr = Counter(zip(true_df.subject, true_df.predicate, true_df.object))
        self.exact_false_ctr = Counter(zip(false_df.subject, false_df.predicate, false_df.object))

        # (subject,predicate) -> Counter of TRUE / FALSE objects seen
        self.sp_true_objs = defaultdict(Counter)
        for s, p, o in zip(true_df.subject, true_df.predicate, true_df.object):
            self.sp_true_objs[(s, p)][o] += 1
        self.sp_false_objs = defaultdict(Counter)
        for s, p, o in zip(false_df.subject, false_df.predicate, false_df.object):
            self.sp_false_objs[(s, p)][o] += 1

        # (predicate,object) -> count TRUE / FALSE elsewhere
        self.po_true_ctr = Counter(zip(true_df.predicate, true_df.object))
        self.po_false_ctr = Counter(zip(false_df.predicate, false_df.object))

        rel_stats = labelled_df.groupby('predicate')['truth_value'].agg(['mean', 'count'])
        self.pred_prior = rel_stats['mean'].to_dict()
        self.pred_n = rel_stats['count'].to_dict()
        self.global_prior = labelled_df['truth_value'].mean()

        g_subj = labelled_df.groupby('subject')['truth_value'].agg(['sum', 'count'])
        self.subject_true_sum = g_subj['sum'].to_dict()
        self.subject_n = g_subj['count'].to_dict()
        g_obj = labelled_df.groupby('object')['truth_value'].agg(['sum', 'count'])
        self.object_true_sum = g_obj['sum'].to_dict()
        self.object_n = g_obj['count'].to_dict()

    def featurize(self, df, exclude_self=False):
        rows = []
        it = zip(df.index, df['subject'], df['predicate'], df['object'],
                  df['truth_value'] if 'truth_value' in df.columns else [None] * len(df))
        for idx, s, p, o, y_self in it:
            key = (s, p, o)
            self_is_true = exclude_self and (y_self == 1)
            self_is_false = exclude_self and (y_self == 0)

            po_true = self.po_true_ctr.get((p, o), 0) - (1 if self_is_true else 0)
            po_false = self.po_false_ctr.get((p, o), 0) - (1 if self_is_false else 0)
            po_true = max(po_true, 0); po_false = max(po_false, 0)

            exact_true = self.exact_true_ctr.get(key, 0) - (1 if self_is_true else 0)
            exact_false = self.exact_false_ctr.get(key, 0) - (1 if self_is_false else 0)
            exact_true = max(exact_true, 0); exact_false = max(exact_false, 0)

            true_objs = self.sp_true_objs.get((s, p), Counter()).copy()
            false_objs = self.sp_false_objs.get((s, p), Counter()).copy()
            if self_is_true and true_objs.get(o, 0) > 0:
                true_objs[o] -= 1
            if self_is_false and false_objs.get(o, 0) > 0:
                false_objs[o] -= 1
            other_true_ct = sum(c for obj, c in true_objs.items() if obj != o)
            other_false_ct = sum(c for obj, c in false_objs.items() if obj != o)

            subj_sum = self.subject_true_sum.get(s, 0.0) - (1 if self_is_true else 0)
            subj_n = self.subject_n.get(s, 0) - (1 if exclude_self and y_self in (0, 1) else 0)
            obj_sum = self.object_true_sum.get(o, 0.0) - (1 if self_is_true else 0)
            obj_n = self.object_n.get(o, 0) - (1 if exclude_self and y_self in (0, 1) else 0)

            pred_prior_n = self.pred_n.get(p, 0) - (1 if exclude_self and y_self in (0, 1) else 0)
            pred_sum = self.pred_prior.get(p, self.global_prior) * self.pred_n.get(p, 0) - (1 if self_is_true else 0)
            predicate_prior = (pred_sum / pred_prior_n) if pred_prior_n > 0 else self.global_prior

            rows.append({
                'exact_true_count': exact_true,
                'exact_false_count': exact_false,
                'po_true_count': po_true,
                'po_false_count': po_false,
                'po_true_minus_false': po_true - po_false,
                'other_true_obj_count_same_sp': other_true_ct,
                'other_false_obj_count_same_sp': other_false_ct,
                'has_other_true_object_same_sp': float(other_true_ct > 0),
                'has_other_false_object_same_sp': float(other_false_ct > 0),
                'subject_true_rate': (subj_sum / subj_n) if subj_n > 0 else np.nan,
                'subject_n': max(subj_n, 0),
                'object_true_rate': (obj_sum / obj_n) if obj_n > 0 else np.nan,
                'object_n': max(obj_n, 0),
                'predicate_prior': predicate_prior,
            })
        feat = pd.DataFrame(rows, index=df.index)
        feat['subject_true_rate'] = feat['subject_true_rate'].fillna(feat['predicate_prior'])
        feat['object_true_rate'] = feat['object_true_rate'].fillna(feat['predicate_prior'])
        return feat
