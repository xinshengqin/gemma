# mSOP-765k offline pipeline — decisions made

Context for reviewing `convert_msop765k.py` and `inspect_msop765k.py`. Records
what was decided and why, plus the choices still open.

Source: [mSOP-765k](https://www.msop-765k.org/) (Lamm & Keuper, TMLR 01/2026),
built on Retail-786k.

## Scope

Offline only: raw source → Bagz. The online preprocessor (parsing, tokenizing,
prompt assembly at train time) is explicitly out of scope, so this directory
contains no grain transforms, kauldron configs, or data sources.

Deliverables: `convert_msop765k.py` (source → Bagz) and `inspect_msop765k.py`
(Bagz → HTML preview of `k` records, default 5).

## Container and record format

| Decision | Rationale |
|---|---|
| Bagz of `tf.train.Example`, all-bytes features | Matches the existing precedent in `gemma/diffusion/hackable_diffusion_adapter/data/sudoku/convert_sudoku.py`, so output is readable by the repo's `PicklableBagzReader` with no new machinery. |
| Image bytes stored **in** the record | Avoids materializing 765k loose JPEGs (inode pressure, slow random reads). |
| Sharded output, `msop765k_{split}-NNNNN-of-NNNNN.bagz` | `bagz.Writer` writes one shard; `bagz.Reader` reads a collection via the `name@N.bagz` spec. Required at full scale (~69 GB train). |
| Shard count auto-derived from `--records_per_shard` | Keeps `--max_records` the *only* knob that differs between a toy and a full run. |
| **Default (auto) compression, not `CompressionNone`** | See "Bug found by the toy run" below. |

### Feature layout

`id`, `product_id`, `filename`, `image/encoded`, `image/format`,
`prompt_system`, `prompt`, `target_json`, plus one feature per target key.

- **`prompt` is stored per record** (requested explicitly). It is constant
  today, so this is a "beta" prompt; every field is also stored individually so
  an online preprocessor can rebuild a different prompt without re-converting.
  Cost is ~300 B against a ~95 KB image (~0.3%).
- **Per-field features are kept alongside `target_json`** so the target view can
  change without re-running the pipeline. ~200 B overhead per record.
- **`product_id`** renames the source's `label` column, which otherwise collides
  with the ML sense of "label". It is the product-cluster id.

## Source handling

| Decision | Rationale |
|---|---|
| Raw mirror separate from conversion | Conversion is then pure, offline, deterministic and re-runnable. The mirror is an incremental cache: shards fetched for a small run are reused by larger ones. |
| Images streamed from `{label}.tar.gz` | No intermediate image files are ever written. Only the mirror and the Bagz output exist on disk. |
| Atomic downloads (`.part` + `os.replace`) | An interrupted run cannot leave a truncated shard that later looks cached. |
| **512px, hardcoded, no flag** | mSOP-765k publishes only 512px and 256px; 512 is the maximum available and what the paper evaluates on. True originals live upstream in Retail-786k. |

Note on cost: a shard holds *all* rows for one label, and gzip cannot be
partially extracted, so download volume scales with **labels touched, not
records kept**. Test shards average ~11 images (~1 MB); train shards ~226
(~21 MB). This is why development targets the test split.

## Sampling

Uniform random sample over records with a fixed seed — **no stratification and
no filtering**. A toy run is therefore an unbiased preview of the full split;
partial label coverage is expected and accepted. Verified: toy(200) field
fill-rates all fall inside the 2σ binomial band of the full 36,571.

Records are written grouped by label (each tarball is opened once), so
`inspect_msop765k.py` samples random indices rather than taking a prefix.

## Normalizations

All are in `normalize_row`, which is pure and unit-testable.

| Field | Rule | Why |
|---|---|---|
| `product_weight` | split into `weight_number` + `weight_unit` | The paper issues these as two separate queries; its metric re-concatenates them. |
| `different_types` | **NaN → `"no"`** | Paper Table 1 reports this target at 100% coverage with a yes/no enum, while the parquet stores "no" as NaN. **Inferred, not verified** — see open issues. |
| `relative_discount` | integer string (`"13"`, not `"13.0"`) | The paper's metric compares it as `str(int(float(x)))`. |
| `GTINs`, `product_category` | split on `", "` into lists | Both are `", "`-joined in the parquet and `List[str]` in the paper's schema. |
| `brand` | **not** split | Brands legitimately contain commas ("Nescafé, Dolce Gusto"). |
| GTINs | kept as strings | 14-digit codes have significant leading zeros ("04012839567131"). |
| numerics | kept as strings | The paper's metric is exact string match, so formatting is locked at write time. |
| missing | JSON `null`; empty bytes in per-field features | — |

## Bug found by the toy run

Shards were first written with `CompressionNone`, reasoning that JPEG is already
compressed. They were unreadable: `bagz.Reader(path)` raised
`INVALID_ARGUMENT: Invalid compressed data`, because a no-compression shard
requires *every* reader to pass matching options — which would have broken the
repo's own `PicklableBagzReader`, which calls `bagz.Reader(path)` bare.

Measured overhead of default compression on incompressible data: **46 bytes per
200 KB (0.02%)**. Switched to plain `bagz.Writer(path)`, matching
`convert_sudoku.py`. Interoperability beats a 0.02% size saving.

## What is not checked in

`mirror/`, `out/` and `reference/` are gitignored: they are large, and they are
downloaded artifacts rather than source.

## The answer: `target_json`

`target_json` is the response an SFT model should generate for the stored
`prompt`. It carries **8 keys = 7 targets** — exactly the targets the paper
scores in its zero-shot evaluation (Tables 4 and 5):

```
brand · weight_number · weight_unit · different_types
price · regular_price · relative_discount · absolute_discount
```

`product_category` and `GTINs` are deliberately excluded. The paper states both
are not present in the advertisement images, generates no zero-shot prediction
for either, and reports no score for them; training a prompt-only model to emit
them from pixels teaches confident invention. There is no flag for this — the
grounded view is the only output.

Two consequences worth knowing:

* Key order matches Table 5's cumulative union, which accumulates targets left
  to right, so the metric can be computed in the stored order.
* Both excluded fields remain as **per-field features** on every record, so a
  different view can be rebuilt by an online preprocessor with no re-conversion
  and no re-download.

The 9 / 10 / 7 counts in the source, this file, and the paper all describe the
same data: the parquet has 9 target columns; `product_weight` splits into
`weight_number` + `weight_unit`, giving 10 field keys; dropping the 2 lookup
fields leaves 8 keys, which is 7 targets.

## Golden test

The guiding principle is that the test is **a production run at the smallest
possible scale**, not a reimplementation of one. `convert_msop765k_test.py`
invokes `convert.main`, the same entry point the CLI uses, against a mirror
holding one image shard in the real on-disk layout, and compares the Bagz file
that comes out against `testdata/golden.bagz`.

`testdata/` holds the miniature mirror and the golden:

```
testdata/golden.bagz                              expected output
testdata/mirror/test.parquet                      one real row, source schema
testdata/mirror/rpp-765k_512/test/10068.tar.gz    the real shard, byte for byte
```

Comparison is **on record content, not bytes**: both files are decoded and
checked record by record for an identical feature set and identical feature
values. The order features are stored in is not part of the contract. A byte
comparison would additionally pin the bagz container and its compression, so a
dependency upgrade could fail it while the data is unchanged, and it would
report only that two binaries differ.

Separately, records are serialized with protobuf's deterministic mode, so
converting the same mirror twice produces identical files. `tf.train.Example`
holds its features in a map, and protobuf orders map entries differently in
every process, so without it a re-run of a conversion yields different bytes for
identical data — which would defeat content-addressing a generated dataset,
verifying a copy, or diffing two conversions. This is a property of the pipeline
worth having on its own; the test does not depend on it.

* Nothing is stubbed. `ensure_parquet` and `ensure_shards` run for real and take
  their cache-hit path because the mirror is pre-populated — the same path every
  rerun of a real conversion takes.
* The fixture is a real subset of the source, never fabricated: one row lifted
  verbatim from `test.parquet` and its label's actual shard copied byte for
  byte. The row is `10068/34468.jpg`, which happens to hit every awkward case at
  once — brand `Herta, Genuss Momente`, four GTINs, commas in
  `product_category`, a NaN `different_types`, a float discount, and two absent
  promotion fields.
* Failure messages print text features in full — the prompt and `target_json`
  are the likeliest things to drift, and a digest of them says nothing. Only
  undecodable payloads such as the image collapse to a digest.
* A separate case asserts the metadata sidecar `main` writes, which the golden
  does not cover; another asserts `_TARGET_KEYS` still governs `target_json`.
* Regenerate after an intended change with `--update_golden`, then review the
  reported diff.

Verified the comparison actually bites, across all five drift modes: a changed
text feature, changed image bytes, a removed feature, an added feature, and a
changed record count.

What one record cannot cover, and stays untested here: the download branches of
`ensure_parquet` and `ensure_shards`, the `--max_records` sampling path, and
multi-shard output. All three only engage above this scale.

## Open issues

These are known divergences and unresolved choices, not defects to fix blindly.

1. **Prompt does not match the released zero-shot code.** The stored prompt
   follows the paper's typeset text; the runner in `run_vlm_commerical.py`
   sends `"Extract all targets."` with 4-space runs (from Python line
   continuations) and a stray double period. The authors' own artifacts
   disagree three ways: `"Extract all targets."` (zero-shot, fine-tuning),
   `"Extract all features."` (RAG), `"Extract the features."` (paper §4).
2. **Deliberate deviation: `null`, not `NaN`.** The paper's prompt says "return
   NaN" while its schema serializes an absent value to `null`. The stored
   prompt says `null` so instruction and supervision agree. This is a knowing
   divergence from the published wording.
3. **Key order** follows Figure 2b (product fields, then promotion); the
   paper's schema declares a different order. Matters for SFT, not for scoring.
4. **Types.** All values are strings here; the paper's schema is typed
   (`float`, `int`, enums).
5. **Nullability.** Every field may be null here. The paper's schema makes
   brand, price, weight and different_types *required*.
6. **`different_types` NaN→"no" is unverified** against the authors'
   evaluation notebook.
7. **Comparability is not settled by the data alone.** Matching Tables 4 and 5
   also requires the full 36,571-row test split and a faithful implementation
   of the paper's `custom_acc` and `⋃_test` metrics. Neither is in this
   directory; the eval side is where the remaining fidelity risk sits.

## Verified on the toy run

`--split test --max_records 200` → 200 records, 195 labels, 1 shard, 16 MB;
25 s cold, 0.7 s warm. Checks: record count; every id joins back to
`test.parquet`; ids unique; all images decode as JPEG with longer edge 512;
`target_json` round-trips against the parquet with zero mismatches; each
normalization behaves as specified; prompt constant across records; key set
matches the schema; sharded spec and random access stable; sample fill-rates
within 2σ of the full split.

`doc/msop765k_test_samples.html` is the checked-in render of 5 sampled records
from that run, regenerated with `inspect_msop765k.py --k 5`.
