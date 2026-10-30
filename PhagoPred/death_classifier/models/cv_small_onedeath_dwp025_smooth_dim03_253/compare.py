"""Out of fold results of this run vs the 188 cell run, by group of cells.

Groups: the original 188 cells (both runs, so the effect of the extra
training data), new random cells, new targeted cells (harder than average,
so not comparable to the others), and all cells.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path('temp/death_classifier')
NEW = RUNS / 'cv_small_onedeath_dwp025_smooth_dim03_253'
OLD = RUNS / 'cv_small_onedeath_dwp025_smooth_dim03'
labels = json.loads((NEW / 'labels' / 'labels.json').read_text())
truth = {f"{f.split('.')[0]}_{c}": v
         for f, cs in labels.items() for c, v in cs.items()}
targeted = {f"{f.split('.')[0]}_{c}"
            for f, cs in json.loads((NEW / 'labels' /
                                     'labels_targeted.json').read_text()).items()
            for c in cs}


def load(run):
    p = pd.read_csv(run / 'cv_last_predictions.csv')
    p = p[~p['bad'].astype(bool)].set_index('cell')
    first = {c: json.loads(Path(f'temp/death_crops_smooth/{c}.json').read_text())
             ['first_frame'] for c in p.index}
    p['first'] = pd.Series(first)
    # Current labels, clipped to the track start as in evaluation
    p['t'] = [np.nan if truth[c] is None else max(truth[c], first[c])
              for c in p.index]
    return p


def metrics(p):
    t, q = p['t'], p['predicted']
    both = t.notna() & q.notna()
    alive_ok = t.isna() & q.isna()
    err = (q - t).abs()
    return {
        'cells': len(p),
        'dead': int(t.notna().sum()),
        'within_5': f"{int(((err <= 5) & both).sum() + alive_ok.sum())} "
                    f"({(((err <= 5) & both).sum() + alive_ok.sum()) / len(p):.0%})",
        'within_20': f"{int(((err <= 20) & both).sum() + alive_ok.sum())} "
                     f"({(((err <= 20) & both).sum() + alive_ok.sum()) / len(p):.0%})",
        'alive_to_dead': int((t.isna() & q.notna()).sum()),
        'dead_to_alive': int((t.notna() & q.isna()).sum()),
        'later_called_at_start':
        int((t.notna() & (t > p['first'] + 5) & (q == p['first'])).sum()),
        'median_error': float(err[both].median()),
    }


new, old = load(NEW), load(OLD)
original = sorted(set(old.index))
added = [c for c in new.index if c not in set(old.index)]
rows = {
    'original 188, old run': metrics(old.loc[original]),
    'original 188, new run': metrics(new.loc[[c for c in original
                                              if c in new.index]]),
    'new random': metrics(new.loc[[c for c in added if c not in targeted]]),
    'new targeted': metrics(new.loc[[c for c in added if c in targeted]]),
    'all, new run': metrics(new),
}
table = pd.DataFrame(rows).T
print(table.to_string())
table.to_csv(NEW / 'comparison.csv')
