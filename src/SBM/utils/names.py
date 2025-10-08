import pandas as pd

def build_name_map(path_csv: str) -> dict:
    df = pd.read_csv(path_csv)
    df = df.fillna("")
    out = {'fd': {}, 'understat': {}, 'fbref': {}}
    for _, r in df.iterrows():
        canon = r['canonical']
        if r['fd_name']:
            out['fd'][r['fd_name']] = canon
        if r['understat_name']:
            out['understat'][r['understat_name']] = canon
        if r['fbref_name']:
            out['fbref'][r['fbref_name']] = canon
    return out

def canonicalize(name: str, source_map: dict, source: str) -> str:
    if not isinstance(name, str):
        return name
    m = source_map.get(source, {})
    return m.get(name, name)
