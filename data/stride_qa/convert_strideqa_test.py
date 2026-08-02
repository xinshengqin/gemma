# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Golden end-to-end tests for the STRIDE-QA conversion pipeline.

Each golden test is a production run at the smallest possible scale: it
invokes `convert.main`, the same entry point the CLI uses, against a mirror
holding real source records in the real on-disk layout, and compares every
Bagz file that comes out against the checked-in goldens.

The comparison is on record content, not bytes: each record must carry the
same set of features, and each feature the same values. The order features
happen to be stored in is not part of the contract.

Nothing about the pipeline is stubbed or reimplemented. The only departure
from a full run is that the mirror is pre-populated, so every ensure_*
function takes its cache-hit path and no network call is made — which is the
same path every rerun of a real conversion takes.

`testdata/mirror/` is a complete miniature mirror holding real data lifted
verbatim from both sources (see `doc/testing.md` for what each fixture is and
why it was picked; the source is CC BY-NC-SA 4.0). Because the fixture is a
real input, the goldens can be reproduced straight from the command line:

    python convert_strideqa.py --source dataset --split val \\
        --category ego_centric_spatial_qa \\
        --mirror_dir testdata/mirror --output_dir /tmp/out

One deliberate deviation from the mSOP-765k testing principles: the
alternation unit test at the bottom feeds `explode_conversations` fabricated
conversations, because the invariant it guards (violations must kill the run)
holds everywhere in the real source, so no real fixture can exercise it. The
deviation is confined to that unit-scope case; every golden runs on real data
only.

Regenerate after an intended change, then review the reported diff:

    python convert_strideqa_test.py --update_golden
"""

import hashlib
import json
import os
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import bagz
import tensorflow as tf

import convert_strideqa as convert

_UPDATE_GOLDEN = flags.DEFINE_bool(
    "update_golden", False, "Rewrite the goldens from this run's output."
)

_TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
_FIXTURE_MIRROR = os.path.join(_TESTDATA, "mirror")
_GOLDEN_DIR = os.path.join(_TESTDATA, "golden")

# One golden per conversion run the pipeline supports: the two dataset
# fixture combos, and the bench run (whose 13 partition files are one run's
# output, compared together).
_RUNS = {
    "dataset_spatial": {
        "source": "dataset", "split": "val", "category": "ego_centric_spatial_qa"
    },
    "dataset_spatiotemporal": {
        "source": "dataset", "split": "val", "category": "ego_centric_spatiotemporal_qa"
    },
    "bench": {"source": "bench"},
}


################################################################################
# MARK: Helpers
################################################################################


def read_records(path: str) -> list[dict[str, tuple[bytes, ...]]]:
  """Decode every record in a Bagz file into its feature mapping.

  Values are tuples because `image/encoded` holds 1 or 4 JPEGs per record.
  """
  reader = bagz.Reader(path)
  records = []
  for index in range(len(reader)):
    example = tf.train.Example()
    example.ParseFromString(reader[index])
    records.append({
        key: tuple(value.bytes_list.value)
        for key, value in example.features.feature.items()
    })
  return records


def _show_one(value: bytes) -> str:
  try:
    text = value.decode("utf-8")
  except UnicodeDecodeError:
    digest = hashlib.sha256(value).hexdigest()[:16]
    return f"<{len(value)} bytes, sha256:{digest}>"
  if len(text) > 400:
    text = f"{text[:400]}… (+{len(text) - 400} chars)"
  return repr(text)


def _show(values: tuple[bytes, ...]) -> str:
  """Render a feature for a failure message.

  Text is shown as text, however long, because the prompt and the target are
  the features most likely to drift and a digest of them says nothing. Only
  undecodable payloads such as the images collapse to a digest.
  """
  if len(values) == 1:
    return _show_one(values[0])
  return "[" + ", ".join(_show_one(value) for value in values) + "]"


################################################################################
# MARK: Golden tests
################################################################################


class GoldenEndToEndTest(absltest.TestCase):

  def _convert(self, run: str, output: str) -> None:
    """Run the production entry point over the fixture mirror.

    The mirror is the checked-in fixture itself, read exactly as a real run
    reads a populated mirror. Nothing is constructed here.
    """
    with flagsaver.flagsaver(
        mirror_dir=_FIXTURE_MIRROR,
        output_dir=output,
        max_records=-1,
        **_RUNS[run],
    ):
      convert.main(["convert_strideqa"])

  def _bagz_names(self, directory: str) -> list[str]:
    return sorted(
        name for name in os.listdir(directory) if name.endswith(".bagz")
    )

  def assertRecordsEqual(self, name, actual, expected) -> None:
    """Every record must carry the same feature set and the same values."""
    self.assertEqual(
        len(actual),
        len(expected),
        f"{name}: record count changed:"
        f" {len(actual)} produced, {len(expected)} golden",
    )
    for index, (got, want) in enumerate(zip(actual, expected)):
      missing = sorted(set(want) - set(got))
      added = sorted(set(got) - set(want))
      self.assertFalse(
          missing or added,
          f"{name} record {index} feature set changed:"
          f" missing={missing} unexpected={added}",
      )
      differing = [key for key in sorted(want) if got[key] != want[key]]
      if differing:
        detail = "\n".join(
            f"  {key}:\n    golden:   {_show(want[key])}"
            f"\n    produced: {_show(got[key])}"
            for key in differing
        )
        self.fail(
            f"{name} record {index} differs from the golden in"
            f" {len(differing)} feature(s):\n{detail}\n"
            "If the change was intended, rerun with --update_golden."
        )

  def _check_against_golden(self, run: str) -> None:
    golden = os.path.join(_GOLDEN_DIR, run)
    if _UPDATE_GOLDEN.value:
      with tempfile.TemporaryDirectory() as tmp:
        self._convert(run, tmp)
        shutil.rmtree(golden, ignore_errors=True)
        os.makedirs(golden)
        for name in self._bagz_names(tmp):
          shutil.copyfile(
              os.path.join(tmp, name), os.path.join(golden, name)
          )
      self.skipTest(f"golden rewritten: {golden}")

    with tempfile.TemporaryDirectory() as tmp:
      self._convert(run, tmp)
      produced_names = self._bagz_names(tmp)
      self.assertEqual(
          produced_names,
          self._bagz_names(golden),
          f"{run}: the set of output Bagz files changed",
      )
      for name in produced_names:
        self.assertRecordsEqual(
            name,
            read_records(os.path.join(tmp, name)),
            read_records(os.path.join(golden, name)),
        )

  def test_dataset_spatial_matches_golden(self):
    self._check_against_golden("dataset_spatial")

  def test_dataset_spatiotemporal_matches_golden(self):
    self._check_against_golden("dataset_spatiotemporal")

  def test_bench_matches_golden(self):
    self._check_against_golden("bench")

  def test_dataset_metadata_sidecar(self):
    """`main` writes the per-combo sidecar, and it describes the shard."""
    with tempfile.TemporaryDirectory() as tmp:
      self._convert("dataset_spatial", tmp)
      stem = "strideqa_val_ego_centric_spatial_qa"
      with open(os.path.join(tmp, f"{stem}.metadata.json")) as handle:
        metadata = json.load(handle)
      records = read_records(
          os.path.join(tmp, f"{stem}-00000-of-00001.bagz")
      )
    self.assertEqual(metadata["split"], "val")
    self.assertEqual(metadata["category"], "ego_centric_spatial_qa")
    self.assertEqual(metadata["num_records"], len(records))
    self.assertEqual(metadata["num_shards"], 1)
    self.assertEqual(metadata["feature_keys"], list(convert._FEATURE_KEYS))
    # The declared key set must still govern what records actually carry;
    # both the golden and the sidecar are pinned to today's output, so the
    # constant could drift while they agree with each other.
    self.assertEqual(sorted(records[0]), sorted(convert._FEATURE_KEYS))

  def test_bench_sidecar_lists_13_specs(self):
    """One bench run must emit all 13 partitions and one accurate sidecar."""
    with tempfile.TemporaryDirectory() as tmp:
      self._convert("bench", tmp)
      with open(
          os.path.join(tmp, "strideqa_bench.metadata.json")
      ) as handle:
        metadata = json.load(handle)
      partitions = metadata["partitions"]
      self.assertLen(partitions, 13)
      self.assertEqual(
          [(p["horizon"], p["qa_category"]) for p in partitions],
          list(convert._BENCH_PARTITIONS),
      )
      for partition in partitions:
        shard = os.path.join(
            tmp,
            f"{os.path.basename(partition['bagz_spec']).split('@')[0]}"
            f"-00000-of-{partition['num_shards']:05d}.bagz",
        )
        records = read_records(shard)
        self.assertLen(records, partition["num_records"])
        for record in records:
          self.assertEqual(
              record["qa_category"],
              (partition["qa_category"].encode(),),
              "records landed in the wrong partition",
          )
      self.assertEqual(
          metadata["num_records"],
          sum(p["num_records"] for p in partitions),
      )


################################################################################
# MARK: Alternation unit test
################################################################################


class ExplodeConversationsTest(absltest.TestCase):
  """Synthetic bad input for the alternation invariant.

  The invariant holds everywhere in the real source, so only fabricated
  conversations can prove a violation kills the run rather than emitting
  misaligned pairs. This is the one deliberate deviation from the
  "never fabricated" fixture principle, confined to unit scope.
  """

  def test_valid_conversation_explodes_in_order(self):
    pairs = convert.explode_conversations([
        {"from": "human", "value": "q0"},
        {"from": "gpt", "value": "a0"},
        {"from": "human", "value": "q1"},
        {"from": "gpt", "value": "a1"},
    ])
    self.assertEqual(pairs, [("q0", "a0"), ("q1", "a1")])

  def test_empty_conversation_yields_no_pairs(self):
    self.assertEqual(convert.explode_conversations([]), [])

  def test_odd_turn_count_is_rejected(self):
    with self.assertRaises(ValueError):
      convert.explode_conversations([{"from": "human", "value": "q"}])

  def test_swapped_roles_are_rejected(self):
    with self.assertRaises(ValueError):
      convert.explode_conversations([
          {"from": "gpt", "value": "a"},
          {"from": "human", "value": "q"},
      ])

  def test_repeated_role_is_rejected(self):
    with self.assertRaises(ValueError):
      convert.explode_conversations([
          {"from": "human", "value": "q0"},
          {"from": "gpt", "value": "a0"},
          {"from": "human", "value": "q1"},
          {"from": "human", "value": "q2"},
      ])


if __name__ == "__main__":
  absltest.main()
