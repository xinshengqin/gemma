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

"""Golden end-to-end test for the mSOP-765k conversion pipeline.

The test is a production run at the smallest possible scale: it invokes
`convert.main`, the same entry point the CLI uses, against a mirror holding one
image shard in the real on-disk layout, and compares the Bagz file that comes
out against `testdata/golden.bagz`.

The comparison is on record content, not bytes: each record must carry the same
set of features, and each feature the same value. The order features happen to
be stored in is not part of the contract.

Nothing about the pipeline is stubbed or reimplemented here. The only departure
from a full run is that the mirror is pre-populated, so `ensure_parquet` and
`ensure_shards` take their cache-hit path and no network call is made — which is
the same path every rerun of a real conversion takes.

`testdata/mirror/` is a complete miniature mirror, holding everything a run
reads and nothing the test invents: the parquet, with the source's own column
order and dtypes, and one image shard in the `{split}/{label}.tar.gz` layout the
downloader produces. It is synthetic, so the test stays offline and no image
from the CC BY-NC-ND source dataset is redistributed here.

Because the fixture is a real input, the golden can be reproduced straight from
the command line:

    python convert_msop765k.py --split test \\
        --mirror_dir testdata/mirror --output_dir /tmp/out

That writes a Bagz file whose records match `testdata/golden.bagz`.

Its single row exercises the awkward cases in the real data: a multi-valued GTIN
list (`04012839567131, 04012839567148`), a brand containing a comma
(`Nescafé, Dolce Gusto`), a NaN `different_types`, a float `relative_discount`
of `13.0`, a `product_weight` of `120.0 Gramm`, and absent `regular_price` and
`absolute_discount`.

Regenerate after an intended change, then review the reported diff:

    python convert_msop765k_test.py --update_golden
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

import convert_msop765k as convert

_UPDATE_GOLDEN = flags.DEFINE_bool(
    "update_golden", False, "Rewrite the golden Bagz from this run's output."
)

_TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
_GOLDEN_PATH = os.path.join(_TESTDATA, "golden.bagz")
_FIXTURE_MIRROR = os.path.join(_TESTDATA, "mirror")

_SPLIT = "test"


################################################################################
# MARK: Helpers
################################################################################


def read_records(path: str) -> list[dict[str, bytes]]:
  """Decode every record in a Bagz file into its feature mapping."""
  reader = bagz.Reader(path)
  records = []
  for index in range(len(reader)):
    example = tf.train.Example()
    example.ParseFromString(reader[index])
    records.append({
        key: value.bytes_list.value[0]
        for key, value in example.features.feature.items()
    })
  return records


def _show(value: bytes) -> str:
  """Render a feature for a failure message.

  Text is shown as text, however long, because the prompt and the target are
  the features most likely to drift and a digest of them says nothing. Only
  undecodable payloads such as the image collapse to a digest.
  """
  try:
    text = value.decode("utf-8")
  except UnicodeDecodeError:
    digest = hashlib.sha256(value).hexdigest()[:16]
    return f"<{len(value)} bytes, sha256:{digest}>"
  if len(text) > 400:
    text = f"{text[:400]}… (+{len(text) - 400} chars)"
  return repr(text)


################################################################################
# MARK: Test
################################################################################


class GoldenEndToEndTest(absltest.TestCase):

  def _convert(self, output: str) -> str:
    """Run the production entry point over the fixture; return the shard.

    The mirror is the checked-in fixture itself, read exactly as a real run
    reads a populated mirror. Nothing is constructed here.
    """
    with flagsaver.flagsaver(
        split=_SPLIT,
        mirror_dir=_FIXTURE_MIRROR,
        output_dir=output,
        max_records=-1,
    ):
      convert.main(["convert_msop765k"])

    shard = os.path.join(output, f"msop765k_{_SPLIT}-00000-of-00001.bagz")
    self.assertTrue(os.path.exists(shard), f"pipeline wrote no shard at {shard}")
    return shard

  def _produced_records(self) -> list[dict[str, bytes]]:
    with tempfile.TemporaryDirectory() as tmp:
      return read_records(self._convert(tmp))

  def assertRecordsEqual(
      self,
      actual: list[dict[str, bytes]],
      expected: list[dict[str, bytes]],
  ) -> None:
    """Every record must carry the same feature set and the same values."""
    self.assertEqual(
        len(actual),
        len(expected),
        f"record count changed: {len(actual)} produced, {len(expected)} golden",
    )
    for index, (got, want) in enumerate(zip(actual, expected)):
      missing = sorted(set(want) - set(got))
      added = sorted(set(got) - set(want))
      self.assertFalse(
          missing or added,
          f"record {index} feature set changed:"
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
            f"record {index} differs from the golden in"
            f" {len(differing)} feature(s):\n{detail}\n"
            "If the change was intended, rerun with --update_golden."
        )

  def test_matches_golden(self):
    if _UPDATE_GOLDEN.value:
      os.makedirs(_TESTDATA, exist_ok=True)
      with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(self._convert(tmp), _GOLDEN_PATH)
      self.skipTest(f"golden rewritten: {_GOLDEN_PATH}")

    self.assertRecordsEqual(self._produced_records(), read_records(_GOLDEN_PATH))

  def test_golden_is_readable_by_a_plain_reader(self):
    """Guards the container itself: no reader-side options may be required."""
    reader = bagz.Reader(_GOLDEN_PATH)
    self.assertLen(reader, 1)
    self.assertNotEmpty(reader[0])

  def test_writes_metadata_alongside_the_shard(self):
    """`main` emits the sidecar describing the run, as production does."""
    with tempfile.TemporaryDirectory() as tmp:
      shard = self._convert(tmp)
      with open(
          os.path.join(os.path.dirname(shard), f"msop765k_{_SPLIT}.metadata.json")
      ) as handle:
        metadata = json.load(handle)
    self.assertEqual(metadata["num_records"], 1)
    self.assertEqual(metadata["num_shards"], 1)
    self.assertEqual(metadata["split"], _SPLIT)
    self.assertEqual(metadata["target_keys"], list(convert._TARGET_KEYS))

  def test_target_json_excludes_lookup_fields(self):
    """The two fields absent from the image stay out of the answer."""
    target = json.loads(
        self._produced_records()[0]["target_json"].decode("utf-8")
    )
    self.assertNotIn("product_category", target)
    self.assertNotIn("GTINs", target)
    self.assertEqual(tuple(target), convert._TARGET_KEYS)

  def test_lookup_fields_remain_available_as_features(self):
    """They are still carried per-field, so another view needs no re-convert."""
    record = self._produced_records()[0]
    self.assertEqual(record["GTINs"], b"04012839567131, 04012839567148")
    self.assertEqual(record["product_category"], b"Scombermix, Scomber Mix")

  def test_prompt_and_target_agree_on_missing_values(self):
    """The prompt names the same sentinel the target actually uses."""
    record = self._produced_records()[0]
    prompt = record["prompt"].decode("utf-8")
    target = json.loads(record["target_json"].decode("utf-8"))
    self.assertIn("return null", prompt)
    self.assertNotIn("NaN", prompt)
    self.assertIsNone(target["regular_price"])


if __name__ == "__main__":
  absltest.main()
