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

"""On-disk layout of the mSOP-765k mirror.

Kept free of flags and heavy imports so both the converter and the viewer can
share it: importing a module that defines flags into another that defines its
own raises `DuplicateFlagError`.
"""

import os

# Images are taken at 512px, the longest edge published by mSOP-765k. The true
# originals live in the upstream Retail-786k dataset.
IMAGE_DIR = "rpp-765k_512"


def parquet_path(mirror_dir: str, split: str) -> str:
  """Where a split's annotation table lives inside the mirror."""
  return os.path.join(mirror_dir, f"{split}.parquet")


def shard_path(mirror_dir: str, split: str, label: str) -> str:
  """Where one product label's image tarball lives inside the mirror."""
  return os.path.join(mirror_dir, IMAGE_DIR, split, f"{label}.tar.gz")
