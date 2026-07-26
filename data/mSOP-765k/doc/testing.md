# Tests in `data/mSOP-765k/`

Inventory of every test in this directory. **Keep this table in sync when tests
are added, removed or renamed.**

## Principles

1. **The end-to-end test is the production pipeline, only smaller.** It invokes
   the real entry point; scale is the sole difference. Nothing is stubbed,
   mocked or reimplemented.
2. **Tests overlap as little as possible.** A case earns its place only by
   covering behaviour no other case does.
3. **Test data is a complete production input.** Pointing the production CLI at
   `testdata/mirror` reproduces the golden, so the test constructs nothing.
4. **The end-to-end test uses a real subset of the production input**, never
   fabricated data. One real row and its real image shard, taken verbatim from
   the source.
5. **Records are compared on content, not bytes** — same feature set, same value
   per feature. Storage order is not part of the contract, and byte equality
   would pin the container and its compression as well as the data.
6. **The golden changes only with human review.**

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

| Test | Guards | Fails when | Why not covered by the golden |
|---|---|---|---|
| `test_matches_golden` | The whole record: every feature the pipeline emits, compared against `testdata/golden.bagz` by feature set and by value | Any change to the record layout, the prompt, a field normalization, or the record count | — it *is* the golden comparison |
| `test_writes_metadata_alongside_the_shard` | The `msop765k_{split}.metadata.json` sidecar `main` writes: record count, shard count, split, target keys | The sidecar stops being written, or its contents drift from the shard | The golden is the `.bagz` alone; the sidecar is a separate output file it never sees |
| `test_target_json_keys_match_the_declared_constant` | `_TARGET_KEYS` still governs what `target_json` contains | The constant is edited without the emitted JSON following, or vice versa | Both the golden and the sidecar are pinned to today's output, so a constant can drift while they agree with each other |

Three earlier cases were removed as pure overlap: the prompt/target sentinel
check, the per-field lookup-field check, and a plain-reader check. Each asserted
a property of the record that `test_matches_golden` already compares, so under
principle 5 none could fail while the golden passed. Verified by construction
before removal.

## The fixture

`testdata/` is a complete production input plus the expected output. The test
constructs nothing.

```
testdata/golden.bagz                              expected output
testdata/mirror/test.parquet                      one real row, source schema
testdata/mirror/rpp-765k_512/test/10068.tar.gz    the real shard, byte for byte
```

Because the fixture is a real mirror, the golden is reproducible from the
command line:

```bash
python convert_msop765k.py --split test \
    --mirror_dir testdata/mirror --output_dir /tmp/out
```

To see what the test compares rather than infer it from a binary, render both
ends to `testdata/visualization/`:

```bash
scripts/visualize_e2e_test_input_output.sh
```

The row is `10068/34468.jpg`, chosen because one real record happens to hit
every awkward case at once: brand `Herta, Genuss Momente`, whose comma must not
be read as a list separator; four GTINs; commas in `product_category`; a NaN
`different_types`; a float `relative_discount`; and absent `regular_price` and
`absolute_discount`.

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
| Breadth across the real dataset | The fixture is one real record, so field combinations it does not contain go unchecked; the toy-run verification in `decisions_made.md` covers 200 records but is not automated |
