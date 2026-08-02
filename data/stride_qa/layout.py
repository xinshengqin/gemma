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

"""On-disk layout of the STRIDE-QA mirror.

Kept free of flags and heavy imports so both the converter and the viewer can
share it: importing a module that defines flags into another that defines its
own raises `DuplicateFlagError`.

The mirror holds both sources:

    mirror/dataset/{split}/annotations/{category}/annotations-NNNNN.jsonl.gz
    mirror/dataset/{split}/images/{filename}.jpg      loose, range-extracted
    mirror/dataset/{split}/tar_index/images-NNNNN.json
    mirror/dataset/{split}/tar_boundaries.json
    mirror/bench/annotation_files/strideqa_bench_tN.json
    mirror/bench/images/CAM_FRONT/{filename}.jpg
"""

import os

SPLITS = ("train", "val")
CATEGORIES = (
    "ego_centric_spatial_qa",
    "ego_centric_spatiotemporal_qa",
    "object_centric_spatial_qa",
)
BENCH_HORIZONS = ("t0", "t1", "t2", "t3")

# Shard counts of the published source, verified against the HuggingFace tree
# API on 2026-08-02. Image tars are ~1.1 GB each and are never downloaded
# whole; single members are extracted with HTTP range requests.
NUM_IMAGE_SHARDS = {"train": 460, "val": 31}
NUM_ANNOTATION_SHARDS = {
    ("train", "ego_centric_spatial_qa"): 6,
    ("train", "ego_centric_spatiotemporal_qa"): 4,
    ("train", "object_centric_spatial_qa"): 6,
    ("val", "ego_centric_spatial_qa"): 1,
    ("val", "ego_centric_spatiotemporal_qa"): 1,
    ("val", "object_centric_spatial_qa"): 1,
}


def image_shard_name(shard: int) -> str:
  """The remote tar name of one image shard, e.g. ``images-00003.tar``."""
  return f"images-{shard:05d}.tar"


def annotation_path(
    mirror_dir: str, split: str, category: str, shard: int
) -> str:
  """Where one annotation shard lives inside the mirror."""
  return os.path.join(
      mirror_dir,
      "dataset",
      split,
      "annotations",
      category,
      f"annotations-{shard:05d}.jsonl.gz",
  )


def image_path(mirror_dir: str, split: str, filename: str) -> str:
  """Where one range-extracted image lives inside the mirror."""
  return os.path.join(mirror_dir, "dataset", split, "images", filename)


def tar_index_path(mirror_dir: str, split: str, shard: int) -> str:
  """Where the cached member index of one image tar lives."""
  return os.path.join(
      mirror_dir, "dataset", split, "tar_index", f"images-{shard:05d}.json"
  )


def tar_boundaries_path(mirror_dir: str, split: str) -> str:
  """Where the shard -> first-member boundary map of a split lives."""
  return os.path.join(mirror_dir, "dataset", split, "tar_boundaries.json")


def bench_annotation_path(mirror_dir: str, horizon: str) -> str:
  """Where one bench horizon annotation file lives, e.g. for ``t0``."""
  return os.path.join(
      mirror_dir, "bench", "annotation_files", f"strideqa_bench_{horizon}.json"
  )


def bench_image_path(mirror_dir: str, image: str) -> str:
  """Where one bench image lives; `image` is the repo-relative path stored in
  the annotations, e.g. ``images/CAM_FRONT/CAM_FRONT__<hash>.jpg``."""
  return os.path.join(mirror_dir, "bench", image)
