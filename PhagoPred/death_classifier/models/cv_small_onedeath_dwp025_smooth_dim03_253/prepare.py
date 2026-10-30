"""Extract crops for newly labelled cells and freeze the labels for this run.

Labels are restricted to cells in the crop cache (h5 files not on this
machine can't be extracted), and frozen so labelling during the run doesn't
change the folds.
"""
import json
from pathlib import Path

from PhagoPred.death_classifier.extract_crops import cache_name, extract_file

DATA = Path('temp/death_labelling_02_10')
CACHE = Path('temp/death_crops_smooth')
OUT = Path('temp/death_classifier/cv_small_onedeath_dwp025_smooth_dim03_253/labels')

labels = json.loads((DATA / 'deaths.json').read_text())
skipped = json.loads((DATA / 'deaths_skipped.json').read_text())
targeted = json.loads((DATA / 'deaths_targeted.json').read_text())


def cached(file, cell):
    return (CACHE / f'{cache_name(file, int(cell))}.npy').exists()


# Only labelled (not bad track) cells: the death model trains with
# bad_weight 0, so bad tracks aren't used
for file, cells in labels.items():
    missing = sorted(int(c) for c in cells if not cached(file, c))
    if missing and (DATA / file).exists():
        print(f'Extracting {len(missing)} cells from {file}')
        for i in range(0, len(missing), 25):
            extract_file(DATA / file, missing[i:i + 25], CACHE,
                         smooth_window=9, weight=True)

OUT.mkdir(parents=True, exist_ok=True)
frozen = {f: {c: v for c, v in cs.items() if cached(f, c)}
          for f, cs in labels.items()}
frozen_skipped = {f: [c for c in cs if cached(f, c)]
                  for f, cs in skipped.items()}
(OUT / 'labels.json').write_text(json.dumps(frozen, indent=4))
(OUT / 'labels_skipped.json').write_text(json.dumps(frozen_skipped, indent=4))
(OUT / 'labels_targeted.json').write_text(json.dumps(targeted, indent=4))
n = lambda d: sum(len(v) for v in d.values())
print(f'Frozen {n(frozen)}/{n(labels)} labels, {n(frozen_skipped)}/'
      f'{n(skipped)} bad tracks')
