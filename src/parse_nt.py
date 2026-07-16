"""
Parser for the reified N-Triples fact-statements used in the
SWC2017/FactBench-style KG-2022 train/test files.

Each "fact" is reified across several lines that all share the same
statement-subject URI, e.g.:

<.../dataset/3226691> rdf:type rdf:Statement .
<.../dataset/3226691> swc:hasTruthValue "0.0"^^xsd:float .     # only in train
<.../dataset/3226691> rdf:subject   <dbpedia:David_Lee_(basketball)> .
<.../dataset/3226691> rdf:predicate <dbpedia-ont:team> .
<.../dataset/3226691> rdf:object    <dbpedia:Houston_Rockets> .

We parse these into one row per statement.
"""
import re
import pandas as pd

LINE_RE = re.compile(r'^<([^>]+)>\s+<([^>]+)>\s+(.*?)\s*\.\s*$')

def _clean_object(raw):
    raw = raw.strip()
    if raw.startswith('<') and raw.endswith('>'):
        return raw[1:-1]
    # literal, possibly with ^^datatype or @lang
    m = re.match(r'^"(.*)"(\^\^<[^>]+>|@[a-zA-Z-]+)?$', raw)
    if m:
        return m.group(1)
    return raw

def parse_statements(path):
    """Return a DataFrame with columns:
    stmt_id, subject, predicate, object, truth_value (NaN if absent)
    """
    records = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            stmt_uri, pred_uri, obj_raw = m.groups()
            stmt_id = stmt_uri.rsplit('/', 1)[-1]
            rec = records.setdefault(stmt_id, {'stmt_id': stmt_id})
            obj_val = _clean_object(obj_raw)

            if pred_uri.endswith('#type'):
                continue  # always rdf:Statement, not informative
            elif pred_uri.endswith('hasTruthValue'):
                rec['truth_value'] = float(obj_val)
            elif pred_uri.endswith('#subject'):
                rec['subject'] = obj_val
            elif pred_uri.endswith('#predicate'):
                rec['predicate'] = obj_val
            elif pred_uri.endswith('#object'):
                rec['object'] = obj_val

    df = pd.DataFrame(list(records.values()))
    if 'truth_value' not in df.columns:
        df['truth_value'] = pd.NA
    # shorten URIs to local (readable) names for convenience
    def local(u):
        if not isinstance(u, str):
            return u
        return u.rsplit('/', 1)[-1]
    df['subject_name'] = df['subject'].map(local)
    df['object_name'] = df['object'].map(local)
    df['predicate_name'] = df['predicate'].map(local)
    cols = ['stmt_id', 'subject', 'predicate', 'object',
            'subject_name', 'predicate_name', 'object_name', 'truth_value']
    return df[cols]

if __name__ == '__main__':
    import os
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    train = parse_statements(os.path.join(_data_dir, 'KG-2022-train_nt.txt'))
    test = parse_statements(os.path.join(_data_dir, 'KG-2022-test_nt.txt'))
    print('train:', train.shape)
    print(train.head())
    print('test:', test.shape)
    print(test.head())
    print('train truth_value counts:\n', train.truth_value.value_counts(dropna=False))
    print('test truth_value counts:\n', test.truth_value.value_counts(dropna=False))
