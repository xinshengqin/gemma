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

Runs the real pipeline over a synthetic mirror and compares the Bagz file it
produces against `testdata/golden.bagz`, record by record and feature by
feature, so an unintended change to the record layout, the prompt, or any field
normalization fails loudly.

The mirror is synthetic on purpose: the test stays offline and deterministic,
and no image from the CC BY-NC-ND source dataset is redistributed here. The
single row still exercises the awkward cases in the real data — a multi-valued
GTIN list, a brand containing a comma, a NaN `different_types`, a float
discount, and absent promotion fields.

The pipeline's input image is read back out of the golden rather than generated,
so the two cannot drift apart and a Pillow upgrade cannot move the fixture. Only
regenerating the golden from scratch needs an encoder.

Regenerate after an intended change, then review the reported diff:

    python convert_msop765k_test.py --update_golden
"""

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile

from absl import flags
from absl.testing import absltest
import bagz
import pandas as pd
import tensorflow as tf

import convert_msop765k as convert

_UPDATE_GOLDEN = flags.DEFINE_bool(
    "update_golden", False, "Rewrite the golden Bagz from this run's output."
)

_TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
_GOLDEN_PATH = os.path.join(_TESTDATA, "golden.bagz")

_SPLIT = "test"
_LABEL = "10000"
_FILENAME = "119.jpg"

# One row shaped like the real parquet, chosen to cover every normalization.
_ROW = {
    "label": _LABEL,
    "filename": _FILENAME,
    "brand": "Nescafé, Dolce Gusto",
    "price": 1.29,
    "regular_price": None,
    "relative_discount": 13.0,
    "absolute_discount": None,
    "product_category": "Scombermix, Scomber Mix",
    "GTINs": "04012839567131, 04012839567148",
    "product_weight": "120.0 Gramm",
    "different_types": None,
}


################################################################################
# MARK: Bagz helpers
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


def _generate_jpeg() -> bytes:
  """Encode a stand-in advertisement crop. Only used to bootstrap a golden."""
  from PIL import Image  # pylint: disable=g-import-not-at-top

  image = Image.new("RGB", (512, 384), (240, 240, 240))
  for x in range(0, 512, 64):
    for y in range(0, 384, 64):
      if (x // 64 + y // 64) % 2:
        image.paste((90, 110, 140), (x, y, x + 64, y + 64))
  buffer = io.BytesIO()
  image.save(buffer, format="JPEG", quality=90, optimize=False)
  return buffer.getvalue()


def _fixture_image() -> bytes:
  """The image fed to the pipeline, taken from the golden when one exists."""
  if os.path.exists(_GOLDEN_PATH):
    return read_records(_GOLDEN_PATH)[0]["image/encoded"]
  return _generate_jpeg()


def _build_mirror(root: str, image_bytes: bytes) -> pd.DataFrame:
  """Write a one-row parquet and its matching image tarball into `root`."""
  frame = pd.DataFrame([_ROW]).astype({"label": str, "filename": str})
  shard = convert.shard_path(root, _SPLIT, _LABEL)
  os.makedirs(os.path.dirname(shard), exist_ok=True)
  with tarfile.open(shard, "w:gz") as tar:
    info = tarfile.TarInfo(f"{_LABEL}/{_FILENAME}")
    info.size = len(image_bytes)
    tar.addfile(info, io.BytesIO(image_bytes))
  return frame


################################################################################
# MARK: Test
################################################################################


class GoldenEndToEndTest(absltest.TestCase):

  def _run_pipeline(self, destination: str) -> str:
    """Run the real conversion into `destination`; return the shard path."""
    with tempfile.TemporaryDirectory() as tmp:
      mirror = os.path.join(tmp, "mirror")
      frame = _build_mirror(mirror, _fixture_image())
      written, num_shards = convert.write_records(
          frame, mirror, _SPLIT, destination
      )
      self.assertEqual(written, 1)
      self.assertEqual(num_shards, 1)
    return os.path.join(
        destination, f"msop765k_{_SPLIT}-00000-of-00001.bagz"
    )

  def _produced_records(self) -> list[dict[str, bytes]]:
    with tempfile.TemporaryDirectory() as tmp:
      return read_records(self._run_pipeline(tmp))

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
        shutil.copyfile(self._run_pipeline(tmp), _GOLDEN_PATH)
      self.skipTest(f"golden rewritten: {_GOLDEN_PATH}")

    self.assertRecordsEqual(self._produced_records(), read_records(_GOLDEN_PATH))

  def test_golden_is_readable_by_a_plain_reader(self):
    """Guards the container itself: no reader-side options may be required."""
    reader = bagz.Reader(_GOLDEN_PATH)
    self.assertLen(reader, 1)
    self.assertNotEmpty(reader[0])

  def test_target_json_excludes_lookup_fields(self):
    """The two fields absent from the image stay out of the answer."""
    record = self._produced_records()[0]
    target = json.loads(record["target_json"].decode("utf-8"))
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
