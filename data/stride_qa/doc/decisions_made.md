# STRIDE-QA offline pipeline — decisions made

Context for reviewing `convert_strideqa.py` and `inspect_strideqa.py`.
Records what was decided and why, plus the choices still open. The design
follows `data/mSOP-765k/` conventions; deviations from that template are
called out explicitly.

Sources: [STRIDE-QA-Dataset](https://huggingface.co/datasets/turing-motors/STRIDE-QA-Dataset)
and [STRIDE-QA-Bench](https://huggingface.co/datasets/turing-motors/STRIDE-QA-Bench)
(Turing Motors, arXiv 2508.10427), both CC BY-NC-SA 4.0.

## Scope

Offline only: raw source → Bagz. The online preprocessor and the evaluation
metrics are explicitly out of scope. One pipeline serves both repositories
(`--source dataset|bench`): they share imagery conventions, and the bench is
the evaluation counterpart of the dataset's training signal.

## Source handling

| Decision | Rationale |
|---|---|
| **No whole-tar download path exists.** Single images are extracted via HTTP range requests | The Dataset repo is 567 GB; a toy or medium run must never pay for it. Hard requirement — there is deliberately no fallback that would download a shard. |
| Boundary map + per-shard member index, both cached in the mirror | Tar members are lexicographically sorted by filename *across* shards (verified by range-reading headers), so filename → shard is a binary search over one ~4 KB probe per shard; member → offset needs one header walk per shard, ever. |
| Fetch guards: require 206, verify Content-Length, `ustar` magic check, member name must equal the request, bytes must decode as JPEG | A 200 response is the whole 1.1 GB tar (its body is never read). The name check catches a stale cached index; the JPEG check catches offset arithmetic gone wrong. Pax headers (typeflag `x`/`g`) are skipped in every walk. |
| The redirect is resolved **once, without a Range header**; all reads then go straight to the CDN over one keep-alive connection | A CDN URL signed *for a specific range* is bound to it (reuse fails with "Auth failed: invalid range"), but one signed with no Range accepts arbitrary ranges — verified against this repo. One request against the rate-limited origin thus serves a whole ~900-read header walk. Origin is never asked for a body (only a redirect status is accepted there), and a no-Range GET is never sent to the CDN — either would be the whole 1.1 GB tar. |
| Backoff on 429/5xx, honoring `Retry-After`; `HF_TOKEN` honored if set | HuggingFace rate-limits anonymous `resolve` requests; a header walk trips the limit reliably. Waiting is correct at toy scale; a token raises the limit. |
| Annotations and bench images are plain atomic downloads (`.part` + `os.replace`, Content-Length checked) | They are small (MBs); an interrupted run cannot leave a truncated file that later looks cached. |
| Mirror stores **loose extracted JPEGs** | Unlike mSOP (whole shards mirrored, streamed at write time), a range-extracted member arrives alone; storing it loose keeps it re-usable by later runs and by the input-stage viewer. Inode pressure is bounded by records selected, and full-scale runs are out of scope here. |
| `--annotation_shards N` (default 1, -1 = all) | Train annotation shards run to ~600 MB per category; one shard is plenty for toy and medium runs. **Consequence: default train sampling is biased to shard 0** — see `requirements_and_design.md`. Val is single-shard and unbiased; bench always fetches all 4 files. |

## Granularity: one record per QA pair (option B)

Both sources explode to one record per QA pair / question.

* Dataset explosion enforces **strict human→gpt alternation**
  (`conversations[2i].from == "human"`, `[2i+1].from == "gpt"`) and the
  `qa_info[i] ↔ turns (2i, 2i+1)` alignment; any violation kills the run.
  Both invariants held over every record inspected, so a violation means the
  source changed under us, and silently emitting misaligned prompt/answer or
  category assignments would poison training data.
* **Empty conversations emit zero records** — logged, not an error. The
  source really contains them, and not marginally: 44 of 14,092 val
  ego-centric spatial records, and 3,950 of 14,092 (28%) val object-centric
  records.
* Cost: image bytes duplicate ~28× on Dataset, 13× on Bench. Accepted in
  design review; records stay self-contained and shardable.
* **TODO (option A)**: one record per source sample with
  `conversations_json` verbatim would preserve VILA multi-turn packing at 1×
  image cost; the current schema already carries everything it needs.

## Sampling semantics

| Source | `--max_records` counts | Why |
|---|---|---|
| dataset | QA-pair records, uniform over all pairs in the mirrored annotation shards, seeded | A toy run previews the pair distribution, not the sample distribution: long conversations contribute proportionally more pairs, exactly as the full conversion would. |
| bench | **scene groups** (each yields 13 records, one per partition) | Sampling records would tear groups apart; sampling groups keeps every partition covered and the composite metrics (which reduce over a group's horizon sequence) computable on any toy output. 2 groups = 26 records = all 13 partitions exercised. |

Selection order is preserved (selected indices are sorted), so output order
is the stream order of the source — matching mSOP.

## Container and record format

| Decision | Rationale |
|---|---|
| Bagz of `tf.train.Example`, all-bytes features | Matches `data/mSOP-765k` and `convert_sudoku.py`; readable by the repo's `PicklableBagzReader` with no new machinery. |
| Default (auto) compression, not `CompressionNone` | Same interoperability lesson mSOP learned: a no-compression shard breaks bare `bagz.Reader(path)`. |
| Deterministic proto serialization | Re-running a conversion must yield identical bytes; protobuf map order varies per process otherwise. |
| **Uniform 23-key schema across both sources**, empty bytes where n/a | One reader/preprocessor handles both; a record declares its own shape (`horizon` empty ⇒ dataset record). |
| `image/encoded` is a **multi-value** BytesList (1 or 4 JPEGs) | The natural encoding for frame sequences; order is source order, current frame **last** (verified against the official bench inference code's `["prev_3", "prev_2", "prev_1", "current"]` zip). |
| `bbox_json` / `rle_json` / `region_json` sample-level, verbatim, no renumbering or subsetting | The pair's text references `Region [k]` by index into the sample-level lists; any subsetting would break the k ↔ annotation join. Shape quirks ride along untouched (spatial: flat lists + `region` list-of-lists; spatiotemporal: per-frame dicts, and the `region` key is absent altogether → empty bytes). |
| `qa_info_json` is the **pair's** `qa_info[i]` entry | It is the per-pair metadata (category, tokens, numeric state); the sample-level annotations are already carried by the three fields above. |
| No `prompt_system` feature | Neither source defines one; storing an invented constant would misrepresent the source. (mSOP has one because its paper does.) |
| Bench numeric truth as `gt_value_json`/`unit_json` + `group_id`/`horizon` | Everything the official composite metrics join on, without re-parsing `gt` text. |

## Outputs

* Dataset: `strideqa_{split}_{category}-NNNNN-of-NNNNN.bagz` + per-combo
  `strideqa_{split}_{category}.metadata.json`. One combo per run.
* Bench: 13 specs `strideqa_bench_{qa_category}-NNNNN-of-NNNNN.bagz` from a
  single run + one `strideqa_bench.metadata.json` listing all 13 with
  per-partition counts. The 1-vs-13 trade-off table is in
  `requirements_and_design.md`.

## Category correspondences (verified 2026-08-02)

* Bench frame order = dataset frame order = current **last**: confirmed in
  `strideqa_bench/inference/inference.py` (`time_buckets` zip).
* `target_distance_t0` ↔ `ego_distance_data`: template pools identical
  16/16.
* Bench t0 bearing (`target_bearing_angle_data`) ↔ the ego-centric
  *spatial* `ego_angle` category: template pools identical 5/5 after
  `Region [k]` normalization; it has no spatiotemporal counterpart. The
  full mapping table is in `requirements_and_design.md`.

## Inspector

| Decision | Rationale |
|---|---|
| Raw + overlay pair per frame; bbox rectangles + RLE mask outlines labeled `Region [k]` (pycocotools) | The text's region references are only checkable against pixels with the marks drawn; showing raw beside overlay keeps occlusions visible. |
| **Display copies downscaled to ~960 px longest edge**, original dims printed | Deviation from mSOP's verbatim embedding, approved in design review: at 2880×1860 × up to 8 renders per record, verbatim pages would run to hundreds of MB. |
| Multi-frame filmstrip, current frame emphasized | The current frame is the one the question is about; the rest are context. |
| `region: null` renders as an absent value | Spatial records carry `region`; spatiotemporal records omit the key. Printing the string "null" would look like data. |
| Placeholder annotations are skipped, not drawn | Dataset spatiotemporal prev-frames carry an empty bbox (`[[]]`) and an RLE stripped to its tokens (no `size`/`counts`) — only the current frame has real geometry. Bench, by contrast, has a real mask on every frame. |

## Fixtures and goldens

`testdata/mirror/` holds three real records lifted verbatim from the
sources (CC BY-NC-SA 4.0, as the sources are):

* dataset-spatial: one val `ego_centric_spatial_qa` sample (1 frame,
  10 QA pairs — the smallest nonzero conversation the split contains;
  spatial conversations come only in multiples of 10 pairs).
* dataset-spatiotemporal: one val sample with 2 QA pairs (the split's
  minimum) and its 4 frames.
* bench: scene group `g00251` — 13 question records across the four horizon
  files and its 4 frames (the group with the smallest total image bytes,
  2.1 MB of 409 groups; median 4.3 MB).

Annotation fixtures are the real records' lines/entries, verbatim; images
are the real bytes, fetched through the pipeline's own range extraction.
The goldens are heavy — the bench golden alone is ~28 MB because 13 records
each embed the group's 4 JPEGs — which is the record-per-QA-pair duplication
cost made visible at fixture scale, accepted in design review.

See `testing.md` for the test inventory and the one deliberate deviation
from mSOP's "never fabricated" principle (the alternation unit test).

## What is not checked in

`mirror/` and `out/` are gitignored: large, and downloaded artifacts rather
than source.

## Open issues

1. **Dataset spatiotemporal per-frame keys are assumed to follow the bench
   convention** (`prev_3` oldest … `current` last) when the viewer overlays
   frames. Verified for bench against official code; for the Dataset it
   rests on `images[3] == id == image_info.file_path` plus the shared
   imagery pipeline. Affects visualization only — stored records carry the
   source dicts verbatim.
2. **Train shard-0 bias** (see above) is documented, not removed; unbiased
   train sampling needs `--annotation_shards -1`.
3. **Comparability to the official benchmark numbers** additionally needs
   the official metric implementation and Set-of-Marks rendering at
   inference time; neither lives here.

## Verified on the toy run

Run 2026-08-02: `run_dataset.sh 10 val <category>` for all three categories
plus `run_bench.sh 2`, i.e. val × 3 categories × 10 QA pairs + bench × 2
scene groups (26 records across all 13 partitions). Costs: mirror 256 MB
(65 range-extracted val images, 27 tar indices, 12 bench images), output
206 MB, no whole tar ever downloaded. Cold conversion time was dominated by
first-touch header walks (~10–25 min per category at ~4 requests/s across
16 parallel shard walks, resumable when interrupted); warm re-runs convert
in seconds.

Checks, all passing (499 assertions over the 56 records):

* record counts match `--max_records` and every sidecar count matches its
  shard's actual record count (`num_shards`, per-partition counts, spec
  names);
* every `id` joins back to its source record; ids unique; `pair_index`
  consistent with the id suffix;
* `prompt` and `target_text` are byte-identical to the source turns
  (dataset) / `question` and `gt` (bench); `gt_value_json` round-trips;
* every stored frame decodes as JPEG with dimensions matching
  `image_info` (2880×1860); 4 frames per spatiotemporal/bench record in
  source order, current last;
* `qa_category` equals `qa_info[i].category` (dataset) and equals its
  partition's category for every bench record (no cross-partition leaks);
  every sampled bench group covers all 13 partitions;
* sharded-spec (`name@N.bagz`) random access works on every output.

The sampled toy output also surfaced the source quirks now documented
above: the 28% empty-conversation rate in object-centric val, and the
placeholder prev-frame annotations in spatiotemporal records.

`testdata/visualization/` holds the checked-in renders of both ends of the
three test fixtures, regenerated by `visualize_e2e_test_input_output.sh`;
overlay correctness was verified by pixel-diffing a rendered overlay
against its raw frame at the record's scaled bbox coordinates and by eye.
