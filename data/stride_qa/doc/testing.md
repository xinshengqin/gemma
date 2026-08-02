# Tests in `data/stride_qa/`

Inventory of every test in this directory. **Keep this table in sync when
tests are added, removed or renamed.**

## Principles

Inherited from `data/mSOP-765k/doc/testing.md`:

1. **The end-to-end test is the production pipeline, only smaller.** Each
   golden case invokes `convert.main`, the real entry point; scale is the
   sole difference. Nothing is stubbed, mocked or reimplemented.
2. **Tests overlap as little as possible.** A case earns its place only by
   covering behaviour no other case does.
3. **Test data is a complete production input.** Pointing the production CLI
   at `testdata/mirror` reproduces the goldens.
4. **The end-to-end fixtures are a real subset of the production input.**
   Real annotation records, verbatim; real image bytes, fetched through the
   pipeline's own range extraction.
5. **Records are compared on content, not bytes** — same feature set, same
   values per feature. Storage order is not part of the contract, and byte
   equality would pin the container and its compression as well as the data.
6. **The goldens change only with human review** (`--update_golden`, then
   review the diff).

**One deliberate deviation from principle 4:** `ExplodeConversationsTest`
feeds `explode_conversations` fabricated conversations. The invariant it
guards — an alternation violation must kill the run — holds everywhere in
the real source, so no real fixture can exercise it. The deviation is
confined to that unit-scope case.

## Running them

```bash
cd data/stride_qa
python convert_strideqa_test.py                 # all of them, offline
python convert_strideqa_test.py GoldenEndToEndTest.test_bench_matches_golden
python convert_strideqa_test.py --update_golden # rewrite goldens, then review
```

They need no network and no mirror: everything they read is checked in under
`testdata/`.

## `convert_strideqa_test.py`

| Test | Guards | Fails when | Why not covered by another case |
|---|---|---|---|
| `test_dataset_spatial_matches_golden` | The full record content of a spatial (1-frame) conversion: explosion, uniform schema, verbatim JSON ride-alongs | Any change to the record layout, a feature value, or the record count | The other goldens exercise 4-frame records and the bench path, not the flat-list spatial annotation shapes |
| `test_dataset_spatiotemporal_matches_golden` | Same, for a 4-frame sample: multi-value `image/encoded`, per-frame dict annotations, absent `region` key → empty | As above, for the spatiotemporal shapes | Spatial golden cannot see 4-frame handling |
| `test_bench_matches_golden` | One bench run's full output: 13 partition files, group sampling, horizon derivation, bench field mapping | As above, for the bench path | Dataset goldens share no code path past `write_bagz` |
| `test_dataset_metadata_sidecar` | The per-combo sidecar `main` writes, and that `_FEATURE_KEYS` still governs the records | The sidecar stops being written, drifts from the shard, or the key-set constant drifts from the output | The goldens are the `.bagz` alone; sidecars are separate files they never see. The constant could drift while golden and sidecar agree with each other |
| `test_bench_sidecar_lists_13_specs` | The single bench sidecar: exactly 13 partitions in the declared order, per-partition counts match the shards, records landed in their own partition's file | A partition goes missing, counts drift, or records cross partitions | The bench golden compares per-file content but never reads the sidecar |
| `ExplodeConversationsTest` (5 cases) | The explosion invariant, pure-function scope: valid order, empty ⇒ no pairs, odd length / swapped roles / repeated role ⇒ `ValueError` | The alternation check weakens into silent misalignment | Golden fixtures are real data, where the invariant never fires |

## The fixture

`testdata/` is a complete production input plus the expected outputs. The
tests construct nothing. The source data is **CC BY-NC-SA 4.0**
(Turing Motors STRIDE-QA); the fixture records and image bytes below are
verbatim excerpts of it, checked in for testing only.

```
testdata/mirror/dataset/val/annotations/{category}/annotations-00000.jsonl.gz
                                       one real record's line each, verbatim
testdata/mirror/dataset/val/images/*.jpg          its frames, real bytes
testdata/mirror/bench/annotation_files/strideqa_bench_t{0-3}.json
                                       one real scene group's 13 records
testdata/mirror/bench/images/CAM_FRONT/*.jpg      its 4 frames, real bytes
testdata/golden/dataset_spatial/          expected spatial output (1 file)
testdata/golden/dataset_spatiotemporal/   expected 4-frame output (1 file)
testdata/golden/bench/                    expected bench output (13 files)
```

Why these records:

* **spatial** — a 10-pair sample: the smallest nonzero conversation val
  spatial contains (conversations come only in multiples of 10 pairs; 44
  records have zero).
* **spatiotemporal** — a 2-pair sample, the split's minimum, so the golden
  stays small despite embedding 4 frames per record.
* **bench** — scene group `g00251`, the smallest of all 409 groups by total
  image bytes (2.1 MB; median 4.3 MB). Its golden is still ~28 MB: 13
  records each embed the group's 4 JPEGs — the record-per-QA-pair
  duplication cost, made visible at fixture scale.

Verified that the golden comparison actually bites, across six drift modes:
a changed text feature, changed image bytes, a removed feature, an added
feature, a changed record count, and a changed `image/encoded` value count.

Because the fixture is a real mirror, the goldens are reproducible from the
command line:

```bash
python convert_strideqa.py --source dataset --split val \
    --category ego_centric_spatial_qa \
    --mirror_dir testdata/mirror --output_dir /tmp/out
```

To see what the tests compare rather than infer it from binaries:

```bash
scripts/visualize_e2e_test_input_output.sh
```

## Not covered

O(1) fixtures at one scale cannot reach these. Worth knowing before trusting
a green suite:

| Gap | Why |
|---|---|
| The network paths: range extraction, boundary probing, header walks, downloads, 429 backoff | The fixture mirror is pre-populated, so every `ensure_*` takes its cache-hit branch. These paths are exercised by the (manual) toy run instead — see `decisions_made.md`. |
| `--max_records` sampling and the seeded permutations | Selection is a no-op when the fixture holds one sample / one group |
| Multi-shard output and `--records_per_shard` | Fixture record counts always fit one shard |
| `--annotation_shards` beyond 1, and the train split | The fixture mirrors one val shard per category |
| `inspect_strideqa.py` | The viewer has no tests |
| Breadth across the real dataset | Field combinations the three fixtures lack go unchecked; the toy-run verification covers a few dozen records but is not automated |
