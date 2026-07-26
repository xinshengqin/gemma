# Setup

Reproducing this on another machine. A clone gets the code and the test fixture
but **no dataset**: `mirror/`, `out/` and `reference/` are gitignored, because
the source is CC BY-NC-ND 4.0 and runs to ~73 GB.

## Quick start

```bash
cd data/mSOP-765k
scripts/setup.sh              # venv, deps, tests, then 200 test records
scripts/setup.sh -1 test      # ... or the whole test split (~3.7 GB)
scripts/setup.sh 200 train    # ... or a slice of train (~4 GB, see below)
```

The script creates `.venv/`, installs dependencies, runs the golden test, fetches
and converts a slice, and renders both ends to HTML. It is safe to re-run: the
mirror is a cache and nothing is downloaded twice.

## Manual steps

### 1. Environment

Python 3.12. Dependencies: `absl-py`, `bagz`, `numpy`, `pandas`, `pyarrow`,
`pillow`, `tensorflow`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install absl-py bagz numpy pandas pyarrow pillow tensorflow
```

### 2. Verify without any data

The golden test is offline and reads only the checked-in fixture, so it works
immediately after cloning:

```bash
python convert_msop765k_test.py     # 3 tests
```

### 3. Fetch and convert

There is no separate download step: the converter fills the mirror with exactly
what the selected records need, then converts.

```bash
python convert_msop765k.py --split test --max_records 200
python convert_msop765k.py --split test                    # whole split
```

| Target | Records | Mirror | Output | Wall clock |
|---|---|---|---|---|
| toy | 200 (test) | ~200 MB | ~16 MB | under a minute |
| test split | 36,571 | ~3.7 GB | ~3.7 GB | tens of minutes |
| train split | 728,892 | ~69 GB | ~69 GB | hours |

**Downloads scale with labels touched, not records kept.** A shard holds every
row for one label and gzip cannot be partially extracted, so 200 randomly
sampled records pull ~200 shards. Test shards average ~11 images (~1 MB); train
shards ~226 (~21 MB), which is why the same record count costs ~20× more on
train. Budget ~150 GB to hold both mirror and output for the full dataset, or
delete the mirror afterwards.

### 4. Look at it

```bash
scripts/visualize_ppl_input.sh  mirror in.html 5 test
scripts/visualize_ppl_output.sh out/msop765k_test@1.bagz out.html 5
scripts/visualize_e2e_test_input_output.sh      # the test fixture, both ends
```

The input viewer samples only rows whose shard is present, so it works on a
partially filled mirror.

## Reading the output

```python
import bagz, tensorflow as tf

reader = bagz.Reader("out/msop765k_test@1.bagz")   # or a single shard path
example = tf.train.Example()
example.ParseFromString(reader[0])
features = {k: v.bytes_list.value[0]
            for k, v in example.features.feature.items()}
features["image/encoded"]   # JPEG bytes
features["prompt"]          # question
features["target_json"]     # answer
```

`out/msop765k_{split}.metadata.json` records the shard count needed for the
`name@N.bagz` spec, along with the seed and record counts of the run.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no rows … have a mirrored shard` | The input viewer found no downloaded shards; run the converter first. |
| `FileNotFoundError` on a `.tar.gz` | The mirror lacks that label; re-run the converter with the same `--seed` and `--max_records`. |
| `Invalid compressed data` from `bagz.Reader` | A shard was written with non-default compression; see `decisions_made.md`. |
| `DuplicateFlagError` | Something imported `convert_msop765k` into another flag-defining module; import `layout` instead. |
| Downloads are slow | Raise `--download_workers` (default 24). |
