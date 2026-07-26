# Tests in `data/mSOP-765k/`

Inventory of every test in this directory. **Keep this table in sync when tests
are added, removed or renamed.**

## Running them

```bash
cd data/mSOP-765k
python convert_msop765k_test.py            # all of them
python convert_msop765k_test.py GoldenEndToEndTest.test_matches_golden
python convert_msop765k_test.py --update_golden   # rewrite the golden, then review the diff
```

They need no network and no mirror: everything they read is checked in under
`testdata/`.

## `convert_msop765k_test.py` — `GoldenEndToEndTest`

Each case runs `convert.main`, the same entry point the CLI uses, against the
checked-in fixture mirror, then inspects what came out.

| Test | Guards | Fails when |
|---|---|---|
| `test_matches_golden` | The whole record: every feature the pipeline emits, compared against `testdata/golden.bagz` by feature set and by value | Any change to the record layout, the prompt, a field normalization, or the record count |
| `test_golden_is_readable_by_a_plain_reader` | The container: the shard opens with a bare `bagz.Reader(path)` | A writer-side option is introduced that readers must mirror — a `CompressionNone` regression is the concrete case |
| `test_writes_metadata_alongside_the_shard` | The `msop765k_{split}.metadata.json` sidecar `main` writes: record count, shard count, split, target keys | The sidecar stops being written, or its counts or target list drift from the shard |
| `test_target_json_excludes_lookup_fields` | `target_json` carries exactly `_TARGET_KEYS`, in order | `product_category` or `GTINs` leak back into the answer, or the key order changes |
| `test_lookup_fields_remain_available_as_features` | The two excluded fields are still present as per-field features | They get dropped from the record, which would force a re-convert to change target view |
| `test_prompt_and_target_agree_on_missing_values` | The prompt names the same sentinel the target uses (`null`, never `NaN`) | The prompt and the supervised answer disagree about how a missing target is written |

## The fixture

`testdata/` is a complete production input plus the expected output. The test
constructs nothing.

```
testdata/golden.bagz                              expected output
testdata/mirror/test.parquet                      source column order and dtypes
testdata/mirror/rpp-765k_512/test/10000.tar.gz    real shard layout
```

Because the fixture is a real mirror, the golden is reproducible from the
command line:

```bash
python convert_msop765k.py --split test \
    --mirror_dir testdata/mirror --output_dir /tmp/out
```

Its single row covers the awkward cases in the real data: a multi-valued GTIN
list, a brand containing a comma (`Nescafé, Dolce Gusto`), a NaN
`different_types`, a float `relative_discount`, and absent `regular_price` and
`absolute_discount`. The image is synthetic, so no CC BY-NC-ND source imagery is
redistributed here.

## What is comparison, and what is not

Records are compared on **content**: the same set of features, and the same
value for each. The order features are stored in is not part of the contract,
and the files are never compared byte for byte — that would additionally pin the
bagz container and its compression, so a dependency upgrade could fail the suite
while the data is unchanged.

Failure messages print text features in full, since the prompt and `target_json`
are the likeliest things to drift; only undecodable payloads such as the image
collapse to a digest.

## Not covered

One record at one scale cannot reach these. Worth knowing before trusting a
green suite:

| Gap | Why |
|---|---|
| Download paths in `ensure_parquet` and `ensure_shards` | The fixture mirror is pre-populated, so both take their cache-hit branch |
| `--max_records` sampling and the seeded permutation | Selection is a no-op when the split holds one row |
| Multi-shard output and `--records_per_shard` | One record always lands in a single shard |
| `inspect_msop765k.py` | The viewer has no tests |
| Anything about the real dataset | The fixture is synthetic; correctness against the true source was checked by the toy-run verification described in `decisions_made.md`, which is not automated |
