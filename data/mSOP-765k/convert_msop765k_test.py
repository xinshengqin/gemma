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

Runs the real pipeline over a synthetic mirror and compares the resulting Bagz
record against a checked-in golden file, so an unintended change to the record
layout, the prompt, or any field normalization fails loudly.

The mirror is synthetic on purpose: the test stays offline and deterministic,
and no image from the CC BY-NC-ND source dataset is redistributed in this
repository. The row values still exercise the awkward cases in the real data —
a multi-valued GTIN list, a brand containing a comma, a NaN `different_types`,
a float discount, and absent promotion fields.

Regenerate the golden after an intended change:

    python convert_msop765k_test.py --update_golden
"""

import base64
import io
import json
import os
import tarfile
import tempfile

from absl import flags
from absl.testing import absltest
import bagz
import pandas as pd
import tensorflow as tf

import convert_msop765k as convert

_UPDATE_GOLDEN = flags.DEFINE_bool(
    "update_golden", False, "Rewrite the golden file from this run's output."
)

_TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
_GOLDEN_PATH = os.path.join(_TESTDATA, "golden_record.json")
_IMAGE_PATH = os.path.join(_TESTDATA, "synthetic_ad.jpg")

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


def _synthetic_jpeg() -> bytes:
  """The checked-in 512px stand-in for an advertisement crop.

  Read from disk rather than regenerated, so the golden hash does not move when
  the Pillow version changes its JPEG encoding.
  """
  with open(_IMAGE_PATH, "rb") as handle:
    return handle.read()


def _build_mirror(root: str) -> pd.DataFrame:
  """Write a one-row parquet and its matching image tarball into `root`."""
  frame = pd.DataFrame([_ROW]).astype({"label": str, "filename": str})

  shard = convert.shard_path(root, _SPLIT, _LABEL)
  os.makedirs(os.path.dirname(shard), exist_ok=True)
  image_bytes = _synthetic_jpeg()
  with tarfile.open(shard, "w:gz") as tar:
    info = tarfile.TarInfo(f"{_LABEL}/{_FILENAME}")
    info.size = len(image_bytes)
    tar.addfile(info, io.BytesIO(image_bytes))
  return frame


def _record_to_dict(record: bytes) -> dict[str, object]:
  """Decode a Bagz record in full.

  The image is carried verbatim, base64-encoded because JSON cannot hold raw
  bytes, so the golden pins the exact image the pipeline emitted rather than a
  digest of it. This is the synthetic fixture, not dataset imagery.
  """
  example = tf.train.Example()
  example.ParseFromString(record)
  features = {
      key: value.bytes_list.value[0]
      for key, value in example.features.feature.items()
  }
  image = features.pop("image/encoded")
  decoded = {
      key: value.decode("utf-8") for key, value in sorted(features.items())
  }
  decoded["image/encoded.base64"] = base64.b64encode(image).decode("ascii")
  return decoded


class GoldenEndToEndTest(absltest.TestCase):

  def _run_pipeline(self) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
      mirror = os.path.join(tmp, "mirror")
      output = os.path.join(tmp, "out")
      frame = _build_mirror(mirror)

      written, num_shards = convert.write_records(frame, mirror, _SPLIT, output)
      self.assertEqual(written, 1)
      self.assertEqual(num_shards, 1)

      reader = bagz.Reader(
          os.path.join(output, f"msop765k_{_SPLIT}@{num_shards}.bagz")
      )
      self.assertLen(reader, 1)
      return _record_to_dict(reader[0])

  def test_record_matches_golden(self):
    actual = self._run_pipeline()

    if _UPDATE_GOLDEN.value:
      os.makedirs(os.path.dirname(_GOLDEN_PATH), exist_ok=True)
      with open(_GOLDEN_PATH, "w") as handle:
        json.dump(actual, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
      self.skipTest(f"golden rewritten: {_GOLDEN_PATH}")

    with open(_GOLDEN_PATH) as handle:
      expected = json.load(handle)

    self.assertEqual(
        actual,
        expected,
        "Converted record no longer matches the golden file. If the change was"
        " intended, rerun with --update_golden and review the diff.",
    )

  def test_target_json_excludes_lookup_fields(self):
    """The two fields absent from the image stay out of the answer."""
    target = json.loads(self._run_pipeline()["target_json"])
    self.assertNotIn("product_category", target)
    self.assertNotIn("GTINs", target)
    self.assertEqual(tuple(target), convert._TARGET_KEYS)

  def test_lookup_fields_remain_available_as_features(self):
    """They are still carried per-field, so another view needs no re-convert."""
    record = self._run_pipeline()
    self.assertEqual(record["GTINs"], "04012839567131, 04012839567148")
    self.assertEqual(record["product_category"], "Scombermix, Scomber Mix")

  def test_prompt_and_target_agree_on_missing_values(self):
    """The prompt names the same sentinel the target actually uses."""
    record = self._run_pipeline()
    target = json.loads(record["target_json"])
    self.assertIn("return null", record["prompt"])
    self.assertNotIn("NaN", record["prompt"])
    self.assertIsNone(target["regular_price"])


if __name__ == "__main__":
  absltest.main()
