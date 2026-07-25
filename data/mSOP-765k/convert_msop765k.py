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

"""Convert the mSOP-765k dataset to Bagz format.

Mirrors the raw source (parquet + per-label ``.tar.gz`` image shards) into
``--mirror_dir``, then streams it into sharded Bagz files of
``tf.train.Example`` records.

The mirror is an incremental cache: shards fetched for a small ``--max_records``
run are reused by later, larger runs.

Images are taken at 512px (the longest edge published by mSOP-765k; the true
originals live in the upstream Retail-786k dataset).
"""

import collections
import concurrent.futures
import json
import math
import os
import tarfile
import urllib.error
import urllib.request

from absl import app
from absl import flags
from absl import logging
import bagz
import numpy as np
import pandas as pd
import tensorflow as tf

################################################################################
# MARK: Constants
################################################################################

_HF_BASE = (
    "https://huggingface.co/datasets/retail-product-promotion/mSOP-765k"
    "/resolve/main"
)
_IMAGE_DIR = "rpp-765k_512"

# Prompt wording follows the paper (Section 4, "Prompts and Structured Output
# Schemata"), except that a missing target is called `null` rather than `NaN`:
# JSON has no NaN literal, and the paper's own schema serializes an absent value
# to `null` despite its wording. Instructing NaN while supervising null would
# train the model against its own instructions.
_PROMPT_SYSTEM = "You are an assistant for question-answering tasks."
_PROMPT_USER = (
    "Do the user-provided task on the input image. The answer must be provided"
    ' in JSON format. The task is: "Extract the features.". If there is no'
    " information of a target, return null."
)

# Keys serialized into `target_json`: the seven targets the paper evaluates in
# its zero-shot setting (Tables 4 and 5), with product weight split into number
# and unit as the paper does. `product_category` and `GTINs` are deliberately
# absent: neither is present in the advertisement image, so the paper generates
# no zero-shot prediction for them and reports no score. Order matches Table 5's
# cumulative union, which accumulates targets left to right.
_TARGET_KEYS = (
    "brand",
    "weight_number",
    "weight_unit",
    "different_types",
    "price",
    "regular_price",
    "relative_discount",
    "absolute_discount",
)

# Every field kept as its own feature, including the two omitted from
# `target_json`. Keeping them means a different target view can be rebuilt from
# a record without re-running this pipeline.
_FIELD_KEYS = (
    "brand",
    "product_category",
    "GTINs",
    "weight_number",
    "weight_unit",
    "different_types",
    "price",
    "regular_price",
    "relative_discount",
    "absolute_discount",
)

# Columns holding a ", "-joined list. `brand` is deliberately excluded: brands
# legitimately contain commas (e.g. "Nescafé, Dolce Gusto").
_LIST_COLUMNS = ("product_category", "GTINs")

################################################################################
# MARK: Flags
################################################################################

_SPLIT = flags.DEFINE_enum(
    "split", "test", ["test", "train"], "Source split to convert."
)
_MAX_RECORDS = flags.DEFINE_integer(
    "max_records",
    -1,
    "Number of records to emit, sampled uniformly at random without"
    " replacement. Use -1 for the whole split.",
)
_SEED = flags.DEFINE_integer("seed", 0, "Seed for the sampling permutation.")
_MIRROR_DIR = flags.DEFINE_string(
    "mirror_dir", "mirror", "Directory holding the raw source mirror."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "out", "Directory to write the output Bagz shards."
)
_RECORDS_PER_SHARD = flags.DEFINE_integer(
    "records_per_shard", 5000, "Target records per output Bagz shard."
)
_DOWNLOAD_WORKERS = flags.DEFINE_integer(
    "download_workers", 24, "Parallel downloads when filling the mirror."
)

################################################################################
# MARK: Normalization
################################################################################


def _clean(value) -> str | None:
  """Return a stripped string, or None for NaN/empty."""
  if value is None or (isinstance(value, float) and math.isnan(value)):
    return None
  text = str(value).strip()
  return text or None


def _as_list(value) -> list[str] | None:
  text = _clean(value)
  if text is None:
    return None
  return [part.strip() for part in text.split(",") if part.strip()]


def _as_int_str(value) -> str | None:
  """Format a discount percentage as an integer string ("13", not "13.0")."""
  text = _clean(value)
  return None if text is None else str(int(float(text)))


def normalize_row(row) -> dict[str, object]:
  """Map one parquet row to the canonical target dictionary.

  Three source quirks are handled here:
    * ``product_weight`` ("120.0 Gramm") splits into number and unit, matching
      the paper's two separate queries.
    * ``different_types`` is yes/no, never absent: the paper reports it at 100%
      coverage while the parquet leaves "no" as NaN.
    * ``relative_discount`` is compared as an integer string by the paper's
      metric, so it is emitted without a decimal part.
  """
  target: dict[str, object] = {}
  target["brand"] = _clean(row.brand)
  for column in _LIST_COLUMNS:
    target[column] = _as_list(getattr(row, column))

  weight = _clean(row.product_weight)
  if weight is None:
    target["weight_number"], target["weight_unit"] = None, None
  else:
    number, _, unit = weight.partition(" ")
    target["weight_number"], target["weight_unit"] = number, unit or None

  target["different_types"] = _clean(row.different_types) or "no"

  for column in ("price", "regular_price", "absolute_discount"):
    target[column] = _clean(getattr(row, column))
  target["relative_discount"] = _as_int_str(row.relative_discount)
  return target


def render_target_json(target: dict[str, object]) -> str:
  """Serialize the target dict with a stable key order."""
  return json.dumps(
      {key: target[key] for key in _TARGET_KEYS}, ensure_ascii=False
  )


################################################################################
# MARK: Mirror
################################################################################


def _download(url: str, path: str) -> None:
  """Fetch `url` to `path` atomically; a partial file is never left behind."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.{os.getpid()}.part"
  with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
    while chunk := response.read(1 << 20):
      handle.write(chunk)
  os.replace(tmp, path)


def ensure_parquet(mirror_dir: str, split: str) -> str:
  path = os.path.join(mirror_dir, f"{split}.parquet")
  if not os.path.exists(path):
    logging.info("Fetching %s.parquet", split)
    _download(f"{_HF_BASE}/{split}.parquet", path)
  return path


def ensure_shards(mirror_dir: str, split: str, labels: list[str]) -> None:
  """Download any per-label image shards not already mirrored."""
  missing = [
      label
      for label in labels
      if not os.path.exists(shard_path(mirror_dir, split, label))
  ]
  if not missing:
    logging.info("Mirror already holds all %d shards", len(labels))
    return

  logging.info(
      "Fetching %d of %d shards (%d cached)",
      len(missing),
      len(labels),
      len(labels) - len(missing),
  )
  failures: list[tuple[str, str]] = []

  def fetch(label: str) -> None:
    url = f"{_HF_BASE}/{_IMAGE_DIR}/{split}/{label}.tar.gz"
    try:
      _download(url, shard_path(mirror_dir, split, label))
    except (urllib.error.URLError, OSError) as exc:  # network / disk
      failures.append((label, str(exc)))

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_DOWNLOAD_WORKERS.value
  ) as pool:
    for done, _ in enumerate(pool.map(fetch, missing), start=1):
      if done % 200 == 0:
        logging.info("  %d/%d", done, len(missing))

  if failures:
    raise RuntimeError(
        f"{len(failures)} shard downloads failed, e.g. {failures[:3]}"
    )


def shard_path(mirror_dir: str, split: str, label: str) -> str:
  return os.path.join(mirror_dir, _IMAGE_DIR, split, f"{label}.tar.gz")


################################################################################
# MARK: Records
################################################################################


def make_tf_example(features_dict: dict[str, bytes]) -> tf.train.Example:
  """Create a tf.train.Example proto from a dictionary of byte features."""
  tf_features = {
      k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
      for k, v in features_dict.items()
  }
  return tf.train.Example(features=tf.train.Features(feature=tf_features))


def build_features(row, image_bytes: bytes) -> dict[str, bytes]:
  """Assemble the byte features for one advertisement image."""
  target = normalize_row(row)
  features: dict[str, bytes] = {
      "id": f"{row.label}/{row.filename}".encode(),
      "product_id": str(row.label).encode(),
      "filename": str(row.filename).encode(),
      "image/encoded": image_bytes,
      "image/format": b"jpeg",
      "prompt_system": _PROMPT_SYSTEM.encode(),
      "prompt": _PROMPT_USER.encode(),
      "target_json": render_target_json(target).encode(),
  }
  # Per-field copies let an online preprocessor rebuild a different prompt or
  # target view without re-running this pipeline. Absent values are empty.
  for key in _FIELD_KEYS:
    value = target[key]
    if value is None:
      encoded = b""
    elif isinstance(value, list):
      encoded = ", ".join(value).encode()
    else:
      encoded = str(value).encode()
    features[key] = encoded
  return features


def write_records(
    frame: pd.DataFrame, mirror_dir: str, split: str, output_dir: str
) -> tuple[int, int]:
  """Stream selected rows out of the mirrored tarballs into Bagz shards.

  Rows are grouped by label so each ``.tar.gz`` is opened exactly once and no
  image is ever written to disk individually.
  """
  num_records = len(frame)
  num_shards = max(1, math.ceil(num_records / _RECORDS_PER_SHARD.value))
  os.makedirs(output_dir, exist_ok=True)

  wanted: dict[str, dict[str, object]] = collections.defaultdict(dict)
  for row in frame.itertuples():
    wanted[str(row.label)][str(row.filename)] = row

  written = 0
  shard_index = -1
  writer = None
  try:
    for label, members in wanted.items():
      with tarfile.open(shard_path(mirror_dir, split, label), "r:gz") as tar:
        for member in tar:
          name = os.path.basename(member.name)
          row = members.get(name)
          if row is None:
            continue
          image_bytes = tar.extractfile(member).read()
          example = make_tf_example(build_features(row, image_bytes))

          target_shard = min(written // _RECORDS_PER_SHARD.value, num_shards - 1)
          if target_shard != shard_index:
            if writer is not None:
              writer.close()
            shard_index = target_shard
            path = os.path.join(
                output_dir,
                f"msop765k_{split}-{shard_index:05d}-of-{num_shards:05d}.bagz",
            )
            logging.info("Writing %s", os.path.basename(path))
            # Default (auto) compression: JPEG payloads are incompressible, but
            # this keeps shards readable by a plain `bagz.Reader(path)`, which
            # `CompressionNone` does not.
            writer = bagz.Writer(path)
          writer.write(example.SerializeToString())
          written += 1
  finally:
    if writer is not None:
      writer.close()

  if written != num_records:
    raise RuntimeError(
        f"expected {num_records} records but wrote {written}; the mirror and"
        " parquet disagree"
    )
  return written, num_shards


################################################################################
# MARK: Main
################################################################################


def main(argv):
  """Sample the requested split and convert it to Bagz."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  split = _SPLIT.value
  mirror_dir = _MIRROR_DIR.value
  output_dir = _OUTPUT_DIR.value

  frame = pd.read_parquet(ensure_parquet(mirror_dir, split))
  frame = frame.astype({"label": str, "filename": str})
  total = len(frame)

  # Uniform sample over records: no stratification, no filtering, so a small
  # run is an unbiased preview of the full split.
  max_records = _MAX_RECORDS.value
  if 0 <= max_records < total:
    order = np.random.default_rng(_SEED.value).permutation(total)
    frame = frame.iloc[np.sort(order[:max_records])].reset_index(drop=True)
  logging.info("Selected %d of %d records from %s", len(frame), total, split)

  labels = sorted(frame.label.unique())
  ensure_shards(mirror_dir, split, labels)

  written, num_shards = write_records(frame, mirror_dir, split, output_dir)
  logging.info(
      "Wrote %d records across %d shard(s) covering %d labels",
      written,
      num_shards,
      len(labels),
  )

  metadata = {
      "split": split,
      "seed": _SEED.value,
      "max_records": max_records,
      "num_records": written,
      "num_shards": num_shards,
      "num_labels": len(labels),
      "source_records_in_split": total,
      "image_dir": _IMAGE_DIR,
      "bagz_spec": os.path.join(
          output_dir, f"msop765k_{split}@{num_shards}.bagz"
      ),
      "prompt_system": _PROMPT_SYSTEM,
      "prompt": _PROMPT_USER,
      "target_keys": list(_TARGET_KEYS),
      "field_keys": list(_FIELD_KEYS),
      "source": f"{_HF_BASE}/",
  }
  metadata_path = os.path.join(output_dir, f"msop765k_{split}.metadata.json")
  with open(metadata_path, "w") as handle:
    json.dump(metadata, handle, indent=2, ensure_ascii=False)
  logging.info("Wrote %s", metadata_path)


if __name__ == "__main__":
  app.run(main)
