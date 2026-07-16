"""
FactCheckingEngine demo / public API
=====================================
Usage:
    from demo import FactChecker
    fc = FactChecker()                       # trains once on the KG-2022 train file
    fc.check("Barack_Obama", "spouse", "Michelle_Obama")
    fc.check("Barack_Obama", "spouse", "Angela_Merkel")

`check()` accepts either full DBpedia URIs or bare local names
(e.g. "Barack_Obama") for subject/object, and either full predicate URIs
or short names (e.g. "spouse", "birthPlace", ...).
"""
import numpy as np
from evaluate import load_pool
from engine import FactCheckingEngine, PREDICATES

DBPEDIA_RES = 'http://dbpedia.org/resource/'
DBPEDIA_ONT = 'http://dbpedia.org/ontology/'


def _to_entity_uri(name):
    if name.startswith('http://'):
        return name
    return DBPEDIA_RES + name


def _to_predicate_uri(name):
    if name.startswith('http://'):
        return name
    return DBPEDIA_ONT + name


class FactChecker:
    def __init__(self, model_type='logreg', use_kge=False, verbose=True):
        self.train, self.test, self.pool = load_pool()
        self.engine = FactCheckingEngine(model_type=model_type, use_kge=use_kge, seed=0)
        self.engine.fit(self.train, self.pool)
        if verbose:
            print(f"FactChecker ready. Trained on {len(self.train)} labelled facts "
                  f"({len(self.pool.entity2idx)} known entities, "
                  f"{len(self.pool.pred2idx)} relations: {PREDICATES}).")

    def check(self, subject, predicate, obj, explain=True):
        import pandas as pd
        s_uri, p_uri, o_uri = _to_entity_uri(subject), _to_predicate_uri(predicate), _to_entity_uri(obj)
        row = pd.DataFrame([{'subject': s_uri, 'predicate': p_uri, 'object': o_uri}])
        veracity = float(self.engine.predict_veracity(row)[0])

        # Honesty check: if subject and/or object never appear anywhere in the
        # known pool (train+test), the model has zero real evidence about them
        # and its raw score is an unreliable extrapolation. Shrink toward the
        # predicate's base rate and say so, rather than reporting false certainty.
        subj_known = s_uri in self.pool.entity2idx
        obj_known = o_uri in self.pool.entity2idx
        low_confidence = not (subj_known and obj_known)
        if low_confidence:
            prior = self.engine.label_ctx.pred_prior.get(
                p_uri, self.engine.label_ctx.global_prior)
            veracity = 0.5 * veracity + 0.5 * prior

        result = {'subject': subject, 'predicate': predicate, 'object': obj,
                  'veracity': round(veracity, 4), 'low_confidence': low_confidence}

        if explain:
            lc = self.engine.label_ctx
            key = (s_uri, p_uri, o_uri)
            evidence = []
            if lc.exact_true_ctr.get(key, 0) > 0:
                evidence.append("this exact fact already appears as TRUE in the training KG")
            if lc.exact_false_ctr.get(key, 0) > 0:
                evidence.append("this exact fact already appears as FALSE in the training KG")
            other_true = lc.sp_true_objs.get((s_uri, p_uri), {})
            other_true = [o for o in other_true if o != o_uri]
            if other_true:
                evidence.append(f"a different, already-confirmed-true object exists for "
                                 f"({subject}, {predicate}): {[o.rsplit('/',1)[-1] for o in other_true][:3]}")
            po_true = lc.po_true_ctr.get((p_uri, o_uri), 0)
            if po_true > 0:
                evidence.append(f"'{obj}' is already a confirmed-true object of '{predicate}' "
                                 f"for {po_true} other subject(s)")
            if not evidence:
                evidence.append("no direct training-KG evidence found; score is based on "
                                 "entity/predicate popularity patterns only")
            if low_confidence:
                evidence.append("LOW CONFIDENCE: subject and/or object never appear in the "
                                 "known knowledge graph -- score blended toward the "
                                 f"'{predicate}' base rate")
            result['evidence'] = evidence
        return result


if __name__ == '__main__':
    fc = FactChecker()
    examples = [
        # a true fact straight from the training KG
        ("Walt_Whitman", "deathPlace", "Camden,_New_Jersey"),
        # a plausible-looking but false fact (wrong Nobel category)
        ("François_Jacob", "award", "Nobel_Prize_in_Literature"),
        # an unseen, made-up combination (cold start)
        ("Some_Random_Person_Not_In_KG", "birthPlace", "Some_Random_City_Not_In_KG"),
    ]
    print()
    for s, p, o in examples:
        r = fc.check(s, p, o)
        print(f"({s}, {p}, {o})")
        print(f"  veracity = {r['veracity']}")
        for e in r['evidence']:
            print(f"  - {e}")
        print()
