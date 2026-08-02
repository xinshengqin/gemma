# Setup

Reproducing this on another machine. A clone gets the code and the test
fixture but **no dataset**: `mirror/` and `out/` are gitignored. Nothing here
ever downloads the 567 GB source — images are extracted one member at a time
with HTTP range requests.

## Quick start

```bash
cd data/stride_qa
scripts/setup.sh        # venv, deps, offline tests, then a toy run of both sources
```

The script creates `.venv/`, installs dependencies, runs the golden tests
(offline), converts a toy slice of the Dataset (val, 3 categories × 10 QA
pairs) and of the Bench (2 scene groups), and renders both ends of both to
HTML. It is safe to re-run: the mirror is a cache and nothing is fetched
twice.

## Manual steps

### 1. Environment

Python 3.11+. Dependencies: `absl-py`, `bagz`, `numpy`, `pillow`,
`tensorflow`, `pycocotools` (no pandas/pyarrow — there is no parquet here).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install absl-py bagz numpy pillow tensorflow pycocotools
```

### 2. Verify without any data

The golden tests are offline and read only the checked-in fixture:

```bash
python convert_strideqa_test.py     # 3 goldens + 2 sidecar cases + 5 unit cases
```

### 3. Fetch and convert

There is no separate download step: the converter fills the mirror with
exactly what the selected records need, then converts.

```bash
# Dataset: one (split, category) combo per run; --max_records counts QA pairs
python convert_strideqa.py --source dataset --split val \
    --category ego_centric_spatial_qa --max_records 10

# Bench: one run writes all 13 partitions; --max_records counts scene groups
python convert_strideqa.py --source bench --max_records 2
```

Or the wrappers: `scripts/run_dataset.sh [max_records] [split] [category …]`
and `scripts/run_bench.sh [max_records]`.

### What a run costs

| Step | When | Cost |
|---|---|---|
| Annotation shard download | once per (split, category) | val: 26–84 MB; train: ~100–260 MB per shard |
| Bench annotation download | once | 4 files, ~34 MB |
| Tar boundary probe | once per split | one ~4 KB range request per shard (31 val / 460 train) |
| **Tar header walk** | once per image shard touched | ~900 × 512-byte range requests, sequential — **minutes per shard**, and HuggingFace rate-limiting can stretch it (see below). Cached in `mirror/dataset/{split}/tar_index/` forever. |
| Image fetch | once per image | one range request of the member's actual size (~0.6–1.7 MB) |
| Bench image fetch | once per image | plain download, ~1 MB each |

Downloads scale with **records selected, never with the 567 GB source**. The
expensive first-touch cost is the header walk; subsequent runs touching the
same shards reuse the cached indices and fetch only member data.

**Rate limiting:** anonymous requests to HuggingFace `resolve` URLs are
rate-limited, and a header walk trips the limit reliably; the pipeline backs
off and retries (waits are logged). Exporting a HuggingFace token raises the
limit substantially:

```bash
export HF_TOKEN=hf_...
```

### 4. Look at it

```bash
scripts/visualize_dataset.sh output out out.html 5 val ego_centric_spatial_qa
scripts/visualize_dataset.sh input mirror in.html 5 val ego_centric_spatial_qa
scripts/visualize_bench.sh output out bench.html 5
scripts/visualize_e2e_test_input_output.sh      # the test fixture, both ends
```

Every page shows raw + region-overlay pairs per frame (downscaled display
copies; original dimensions printed). The input viewer samples only records
whose images are already mirrored, so it works on a partially filled mirror.

## Reading the output

```python
import bagz, tensorflow as tf

reader = bagz.Reader("out/strideqa_val_ego_centric_spatial_qa@1.bagz")
example = tf.train.Example()
example.ParseFromString(reader[0])
features = {k: list(v.bytes_list.value)
            for k, v in example.features.feature.items()}
features["image/encoded"]     # 1 or 4 JPEGs — a multi-value BytesList
features["prompt"][0]         # question, verbatim
features["target_text"][0]    # answer, verbatim
```

Sidecars record the shard count for the `name@N.bagz` spec plus the run's
seed and counts: `strideqa_{split}_{category}.metadata.json` per dataset
combo, and a single `strideqa_bench.metadata.json` listing all 13 bench
partitions.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `server ignored Range` / `expected 206` | The remote answered 200 or an error; the body is never read. Retry — or the URL scheme changed, in which case do **not** work around it by downloading shards. |
| Long pauses with `HTTP 429 … retrying in N s` | HuggingFace rate limiting during a header walk. Wait it out, or set `HF_TOKEN`. |
| `stale tar_index?` | A cached member index disagrees with the remote tar (republished shard?). Delete that `mirror/dataset/{split}/tar_index/images-NNNNN.json` and re-run. |
| `X not in images-NNNNN.tar; the annotations and the image tars disagree` | Boundary map or index out of date after a source republish; delete `tar_boundaries.json` and the shard's index, re-run. |
| `no … records have all images mirrored` from the input viewer | The mirror lacks those images; run the converter first with the same flags. |
| `conversation has N turns` / `not a human->gpt pair` | The source's alternation invariant broke — the run is meant to die. Investigate the record before touching the check. |
| `DuplicateFlagError` | Something imported `convert_strideqa` into another flag-defining module; import `layout` instead. |
