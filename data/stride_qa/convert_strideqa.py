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

"""Convert STRIDE-QA (Dataset and Bench) to Bagz format.

Mirrors exactly the slice of the raw source that the selected records need
into ``--mirror_dir``, then converts it into sharded Bagz files of all-bytes
``tf.train.Example`` records — one record per QA pair.

The Dataset image tars total 567 GB and are **never downloaded whole**: single
members are extracted with HTTP range requests against the sorted tars (see
`_RangedReader` and `ensure_images`). Bench images are plain per-file
downloads. The mirror is an incremental cache: files fetched for a small
``--max_records`` run are reused by later, larger runs.

Outputs, per run:

  --source dataset   one spec for the selected (--split, --category) combo:
                     ``strideqa_{split}_{category}-NNNNN-of-NNNNN.bagz``
                     plus a per-combo metadata sidecar.
  --source bench     thirteen specs, partitioned by ``qa_category``:
                     ``strideqa_bench_{qa_category}-NNNNN-of-NNNNN.bagz``
                     plus a single ``strideqa_bench.metadata.json``.
"""

import base64
import bisect
import concurrent.futures
import gzip
import http.client
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from absl import app
from absl import flags
from absl import logging
import bagz
import numpy as np
from PIL import Image
import tensorflow as tf

import layout

################################################################################
# MARK: Constants
################################################################################

_DATASET_BASE = (
    "https://huggingface.co/datasets/turing-motors/STRIDE-QA-Dataset"
    "/resolve/main"
)
_BENCH_BASE = (
    "https://huggingface.co/datasets/turing-motors/STRIDE-QA-Bench"
    "/resolve/main"
)

# The 13 bench partitions: every (horizon, qa_category) the four annotation
# files contain. t0 asks distance, speed, bearing and target speed separately;
# t1..t3 ask three questions each, with distance+direction joint (the
# `ego_distance_data_4d_tN` categories carry both quantities despite the name).
_BENCH_PARTITIONS = (
    ("t0", "ego_distance_data"),
    ("t0", "ego_speed_data"),
    ("t0", "target_bearing_angle_data"),
    ("t0", "target_speed_data"),
    ("t1", "ego_distance_data_4d_t1"),
    ("t1", "ego_speed_data_4d_t1"),
    ("t1", "target_speed_data_4d_t1"),
    ("t2", "ego_distance_data_4d_t2"),
    ("t2", "ego_speed_data_4d_t2"),
    ("t2", "target_speed_data_4d_t2"),
    ("t3", "ego_distance_data_4d_t3"),
    ("t3", "ego_speed_data_4d_t3"),
    ("t3", "target_speed_data_4d_t3"),
)

# The uniform key set every record carries, both sources; features that do not
# apply to a source hold empty bytes. `image/encoded` is a BytesList of 1
# (spatial) or 4 (spatiotemporal, bench) JPEGs in source order, current frame
# last — verified against the official bench inference code, which zips
# `images` with ["prev_3", "prev_2", "prev_1", "current"].
_FEATURE_KEYS = (
    "id",
    "image/encoded",
    "image/filenames_json",
    "image/format",
    "prompt",
    "target_text",
    "qa_category",
    "qa_info_json",
    "gt_value_json",
    "unit_json",
    "group_id",
    "horizon",
    "region_id",
    "object_class",
    "bbox_json",
    "rle_json",
    "region_json",
    "image_info_json",
    "token_info_json",
    "sample_id",
    "pair_index",
    "category",
    "split",
)

_TAR_BLOCK = 512

################################################################################
# MARK: Flags
################################################################################

_SOURCE = flags.DEFINE_enum(
    "source", "dataset", ["dataset", "bench"], "Source repository to convert."
)
_SPLIT = flags.DEFINE_enum(
    "split", "val", list(layout.SPLITS), "Dataset split (source=dataset only)."
)
_CATEGORY = flags.DEFINE_enum(
    "category",
    "ego_centric_spatial_qa",
    list(layout.CATEGORIES),
    "Annotation category (source=dataset only).",
)
_ANNOTATION_SHARDS = flags.DEFINE_integer(
    "annotation_shards",
    1,
    "Annotation shards to mirror and sample from, counted from shard 0"
    " (source=dataset only). Use -1 for all. Train sampling is therefore"
    " biased to shard 0 by default; val has a single shard and is unbiased.",
)
_MAX_RECORDS = flags.DEFINE_integer(
    "max_records",
    -1,
    "How much to convert, sampled uniformly at random without replacement:"
    " QA-pair records for source=dataset, scene groups for source=bench (each"
    " group yields 13 records, one per partition). Use -1 for everything in"
    " the mirrored annotations.",
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
    "download_workers", 16, "Parallel image fetches when filling the mirror."
)

################################################################################
# MARK: Ranged HTTP
################################################################################


# Statuses worth waiting out: HuggingFace rate-limits anonymous `resolve`
# requests, and a long header walk trips the limit reliably. The limit
# window is minutes long, so the total backoff must be able to outlast it
# (attempts sum to ~7 minutes).
_TRANSIENT_STATUSES = (429, 500, 502, 503, 504)
_MAX_ATTEMPTS = 8


def _backoff_delay(error: urllib.error.HTTPError, attempt: int) -> float:
  retry_after = (error.headers or {}).get("Retry-After")
  if retry_after and retry_after.isdigit():
    return min(300.0, float(retry_after))
  return min(120.0, 4.0 * 2.0 ** attempt)


def _origin_headers() -> dict[str, str]:
  # A token is never required, but raises the anonymous rate limit.
  token = os.environ.get("HF_TOKEN")
  if token:
    return {"Authorization": f"Bearer {token}"}
  return {}


def _open_https(host: str, port: int) -> http.client.HTTPSConnection:
  """An HTTPS connection to host, tunneled through https_proxy if set.

  http.client does not honor proxy environment variables the way urllib
  does, so behind a proxy the host may not even resolve locally; a CONNECT
  tunnel keeps the connection alive across requests just the same.
  """
  proxy = urllib.request.getproxies().get("https")
  if not proxy or urllib.request.proxy_bypass(host):
    return http.client.HTTPSConnection(host, port)
  parts = urllib.parse.urlsplit(proxy)
  connection = http.client.HTTPSConnection(parts.hostname, parts.port)
  headers = {}
  if parts.username:
    credentials = base64.b64encode(
        f"{parts.username}:{parts.password}".encode()
    ).decode()
    headers["Proxy-Authorization"] = f"Basic {credentials}"
  connection.set_tunnel(host, port, headers=headers)
  return connection


class _RangedReader:
  """Byte-range reader for one remote file over a kept-alive connection.

  HuggingFace ``resolve`` URLs redirect to a signed CDN URL. Resolved
  *without* a Range header, that signature authorizes **arbitrary** ranges
  (verified against this repo; a URL resolved with a Range is bound to that
  exact range and fails on reuse with "Auth failed: invalid range"). So the
  redirect is resolved once — one request against the rate-limited origin —
  and every read is then a ranged GET straight to the CDN over a single
  keep-alive connection. That matters: walking a tar reads one 512-byte
  header per member, and both per-request TLS setup and per-read origin
  round-trips would dominate.

  The origin resolve never reads a body: only a redirect status is accepted
  there (a 200 body would be the whole 1.1 GB tar). Every read enforces the
  fetch guards: the response must be 206 and its Content-Length must equal
  the requested length. A 403 from the CDN means the signature expired; it
  is re-resolved and the read retried.
  """

  _MAX_REDIRECTS = 3

  def __init__(self, url: str):
    parts = urllib.parse.urlsplit(url)
    self._origin_host = parts.hostname
    self._origin_port = parts.port or 443
    self._origin_path = parts.path + (
        f"?{parts.query}" if parts.query else ""
    )
    self._url = url
    self._cdn: http.client.HTTPSConnection | None = None
    self._cdn_parts: urllib.parse.SplitResult | None = None

  def _resolve(self) -> None:
    """One no-Range GET to origin; its redirect is the range-free signed URL.

    Only origin is ever requested without a Range: a no-Range GET to the CDN
    would be answered with the entire remote file. The CDN URL is taken from
    the Location header, unrequested; any further redirect hops are followed
    by `read`, which always carries a Range.
    """
    connection = _open_https(self._origin_host, self._origin_port)
    try:
      connection.request("GET", self._origin_path, headers=_origin_headers())
      response = connection.getresponse()
      if response.status not in (301, 302, 303, 307, 308):
        # A 200 here would be the whole file: never read, carry the status
        # out for the backoff logic.
        raise urllib.error.HTTPError(
            self._url,
            response.status,
            "expected a redirect from origin",
            dict(response.getheaders()),
            None,
        )
      location = urllib.parse.urljoin(
          self._url, response.getheader("Location")
      )
    finally:
      connection.close()
    parts = urllib.parse.urlsplit(location)
    self._cdn = _open_https(parts.hostname, parts.port or 443)
    self._cdn_parts = parts

  def read(self, start: int, length: int) -> bytes:
    """Fetch exactly ``length`` bytes at ``start``, with retries.

    Rate-limit and server-side statuses are waited out with backoff; an
    expired CDN signature (403) is re-resolved; dropped keep-alive
    connections reconnect and retry immediately.
    """
    for attempt in range(_MAX_ATTEMPTS):
      final = attempt == _MAX_ATTEMPTS - 1
      try:
        if self._cdn is None:
          self._resolve()
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        for _ in range(self._MAX_REDIRECTS):
          path = self._cdn_parts.path + (
              f"?{self._cdn_parts.query}" if self._cdn_parts.query else ""
          )
          self._cdn.request("GET", path, headers=headers)
          response = self._cdn.getresponse()
          if response.status not in (301, 302, 303, 307, 308):
            break
          # A rare extra hop; still ranged, so the body stays tiny.
          location = urllib.parse.urljoin(
              f"https://{self._cdn_parts.hostname}{path}",
              response.getheader("Location"),
          )
          response.read()
          parts = urllib.parse.urlsplit(location)
          if (parts.hostname, parts.port) != (
              self._cdn_parts.hostname,
              self._cdn_parts.port,
          ):
            self._cdn.close()
            self._cdn = _open_https(parts.hostname, parts.port or 443)
          self._cdn_parts = parts
        if response.status != 206:
          # Never read the body: a 200 body is the entire remote file.
          raise urllib.error.HTTPError(
              self._url,
              response.status,
              "expected 206",
              dict(response.getheaders()),
              None,
          )
        expected = int(response.getheader("Content-Length", "-1"))
        if expected != length:
          raise RuntimeError(
              f"{self._url}: asked for {length} bytes at {start}, server"
              f" announced {expected}"
          )
        data = response.read()
        if len(data) != length:
          raise RuntimeError(
              f"{self._url}: connection dropped mid-range at {start}"
          )
        return data
      except urllib.error.HTTPError as error:
        self.close()
        if error.code == 403 and not final:
          continue  # expired CDN signature: re-resolve immediately
        if final or error.code not in _TRANSIENT_STATUSES:
          raise
        delay = _backoff_delay(error, attempt)
        logging.info(
            "HTTP %d from %s; retrying in %.0f s",
            error.code,
            self._origin_host,
            delay,
        )
        time.sleep(delay)
      except (http.client.HTTPException, OSError):
        self.close()
        if final:
          raise
    raise AssertionError("unreachable")

  def close(self) -> None:
    if self._cdn is not None:
      self._cdn.close()
    self._cdn = None
    self._cdn_parts = None


################################################################################
# MARK: Remote tar walking
################################################################################


def _parse_tar_header(
    block: bytes, offset: int
) -> tuple[str, int, str] | None:
  """Parse one 512-byte ustar header into (name, size, typeflag).

  Returns None at the end-of-archive zero block.
  """
  if not block.strip(b"\0"):
    return None
  if block[257:262] != b"ustar":
    raise RuntimeError(f"no ustar magic in header at offset {offset}")
  name = block[:100].rstrip(b"\0").decode()
  size = int(block[124:136].rstrip(b"\0 ").decode() or "0", 8)
  typeflag = chr(block[156])
  return name, size, typeflag


def _is_member(name: str, typeflag: str) -> bool:
  """True for a regular file entry; pax headers (`x`/`g`) are metadata."""
  return typeflag in ("0", "\0") and name != "././@PaxHeader"


def _data_end(data_offset: int, size: int) -> int:
  """Offset of the block following a member's data."""
  return data_offset + (size + _TAR_BLOCK - 1) // _TAR_BLOCK * _TAR_BLOCK


def _first_member_name(reader: _RangedReader) -> str:
  """Name of the first regular member, from one probe of the tar's head."""
  probe = reader.read(0, 8 * _TAR_BLOCK)
  offset = 0
  while offset + _TAR_BLOCK <= len(probe):
    parsed = _parse_tar_header(probe[offset : offset + _TAR_BLOCK], offset)
    if parsed is None:
      break
    name, size, typeflag = parsed
    if _is_member(name, typeflag):
      return name
    offset = _data_end(offset + _TAR_BLOCK, size)
  raise RuntimeError("no regular member in the first 4 KiB of the tar")


def _walk_tar_members(
    reader: _RangedReader,
    members: dict[str, tuple[int, int]],
    offset: int,
    checkpoint,
) -> dict[str, tuple[int, int]]:
  """Map member names to (data offset, size) by walking headers from `offset`.

  One 512-byte range request per entry; the data itself is never fetched.
  `checkpoint(members, next_offset)` is called periodically so a walk killed
  by exhausted rate-limit retries can resume instead of starting over.
  """
  entries = 0
  while True:
    parsed = _parse_tar_header(reader.read(offset, _TAR_BLOCK), offset)
    if parsed is None:
      return members
    name, size, typeflag = parsed
    data_offset = offset + _TAR_BLOCK
    if _is_member(name, typeflag):
      members[name] = (data_offset, size)
    offset = _data_end(data_offset, size)
    entries += 1
    if entries % 200 == 0:
      checkpoint(members, offset)
      logging.info("  %d members indexed", len(members))


def _image_shard_url(split: str, shard: int) -> str:
  return f"{_DATASET_BASE}/{split}/images/{layout.image_shard_name(shard)}"


def ensure_boundaries(mirror_dir: str, split: str) -> list[str]:
  """First-member name of every image tar, probed once and cached.

  Members are lexicographically sorted by filename **across** shards
  (verified against the source), so this list maps any filename to its shard
  by binary search.
  """
  path = layout.tar_boundaries_path(mirror_dir, split)
  if os.path.exists(path):
    with open(path) as handle:
      return json.load(handle)["first_members"]

  num_shards = layout.NUM_IMAGE_SHARDS[split]
  logging.info(
      "Probing first members of %d %s image tars", num_shards, split
  )
  first_members = []
  for shard in range(num_shards):
    reader = _RangedReader(_image_shard_url(split, shard))
    try:
      first_members.append(_first_member_name(reader))
    finally:
      reader.close()
    if (shard + 1) % 50 == 0:
      logging.info("  %d/%d", shard + 1, num_shards)
  if first_members != sorted(first_members):
    raise RuntimeError(
        f"{split} image tars are not sorted across shards; the boundary map"
        " cannot locate members"
    )

  _write_atomically(
      path, json.dumps({"first_members": first_members}, indent=2).encode()
  )
  return first_members


def ensure_tar_index(
    mirror_dir: str, split: str, shard: int, reader: _RangedReader
) -> dict[str, tuple[int, int]]:
  """The member index of one image tar, walked once and cached.

  A cache file holding a ``resume_offset`` is a partial walk (killed by
  exhausted rate-limit retries); the walk continues from that offset.
  """
  path = layout.tar_index_path(mirror_dir, split, shard)
  members: dict[str, tuple[int, int]] = {}
  offset = 0
  if os.path.exists(path):
    with open(path) as handle:
      data = json.load(handle)
    members = {k: tuple(v) for k, v in data["members"].items()}
    if "resume_offset" not in data:
      return members
    offset = data["resume_offset"]

  name = layout.image_shard_name(shard)
  logging.info(
      "Walking headers of %s/%s%s",
      split,
      name,
      f" (resuming at {offset})" if offset else "",
  )

  def checkpoint(current: dict[str, tuple[int, int]], next_offset: int):
    _write_atomically(
        path,
        json.dumps({
            "members": {k: list(v) for k, v in current.items()},
            "resume_offset": next_offset,
        }).encode(),
    )

  members = _walk_tar_members(reader, members, offset, checkpoint)
  _write_atomically(
      path,
      json.dumps(
          {"members": {k: list(v) for k, v in members.items()}}
      ).encode(),
  )
  return members


def _fetch_member(
    reader: _RangedReader, name: str, data_offset: int, size: int
) -> bytes:
  """One member's bytes, with every fetch guard applied.

  The request covers the member's header block too, so the name and size can
  be checked against what the (cached) index claims before the payload is
  trusted.
  """
  block = reader.read(data_offset - _TAR_BLOCK, _TAR_BLOCK + size)
  header = _parse_tar_header(block[:_TAR_BLOCK], data_offset - _TAR_BLOCK)
  if header is None or header[0] != name or header[1] != size:
    raise RuntimeError(
        f"tar member at offset {data_offset} is {header}, expected"
        f" ({name!r}, {size}); stale tar_index?"
    )
  data = block[_TAR_BLOCK:]
  with Image.open(io.BytesIO(data)) as image:
    if image.format != "JPEG":
      raise RuntimeError(f"{name}: fetched bytes are {image.format}, not JPEG")
    image.verify()
  return data


def ensure_images(
    mirror_dir: str, split: str, filenames: set[str]
) -> None:
  """Range-extract any images not already mirrored, as loose JPEGs.

  Shards are processed in parallel; within a shard one keep-alive reader
  walks the headers (if the index is not yet cached) and then fetches each
  wanted member.
  """
  missing = sorted(
      name
      for name in filenames
      if not os.path.exists(layout.image_path(mirror_dir, split, name))
  )
  if not missing:
    logging.info("Mirror already holds all %d images", len(filenames))
    return
  logging.info(
      "Fetching %d of %d images (%d cached)",
      len(missing),
      len(filenames),
      len(filenames) - len(missing),
  )

  boundaries = ensure_boundaries(mirror_dir, split)
  by_shard: dict[int, list[str]] = {}
  for name in missing:
    shard = bisect.bisect_right(boundaries, name) - 1
    if shard < 0:
      raise RuntimeError(f"{name} sorts before the first tar member")
    by_shard.setdefault(shard, []).append(name)

  def fetch_shard(shard: int, names: list[str]) -> None:
    reader = _RangedReader(_image_shard_url(split, shard))
    try:
      index = ensure_tar_index(mirror_dir, split, shard, reader)
      for name in names:
        if name not in index:
          raise RuntimeError(
              f"{name} not in {layout.image_shard_name(shard)}; the"
              " annotations and the image tars disagree"
          )
        data_offset, size = index[name]
        data = _fetch_member(reader, name, data_offset, size)
        _write_atomically(layout.image_path(mirror_dir, split, name), data)
    finally:
      reader.close()

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_DOWNLOAD_WORKERS.value
  ) as pool:
    futures = [
        pool.submit(fetch_shard, shard, names)
        for shard, names in sorted(by_shard.items())
    ]
    for future in concurrent.futures.as_completed(futures):
      future.result()


################################################################################
# MARK: Plain downloads
################################################################################


def _write_atomically(path: str, data: bytes) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.{os.getpid()}.part"
  with open(tmp, "wb") as handle:
    handle.write(data)
  os.replace(tmp, path)


def _download(url: str, path: str) -> None:
  """Fetch `url` to `path` atomically; a partial file is never left behind.

  Rate-limit and server-side statuses are waited out with the same backoff
  as ranged reads.
  """
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.{os.getpid()}.part"
  request = urllib.request.Request(url)
  token = os.environ.get("HF_TOKEN")
  if token:
    request.add_header("Authorization", f"Bearer {token}")
  for attempt in range(_MAX_ATTEMPTS):
    try:
      with urllib.request.urlopen(request) as response, open(
          tmp, "wb"
      ) as handle:
        while chunk := response.read(1 << 20):
          handle.write(chunk)
        # A connection dropped mid-body reads as a clean EOF: read(amt)
        # returns b"" instead of raising, so without this check a truncated
        # file would be renamed into the mirror and cached as if complete.
        expected = response.getheader("Content-Length")
        if expected is not None and handle.tell() != int(expected):
          raise urllib.error.ContentTooShortError(
              f"{url}: got {handle.tell()} of {expected} bytes", b""
          )
      os.replace(tmp, path)
      return
    except urllib.error.HTTPError as error:
      if (
          attempt == _MAX_ATTEMPTS - 1
          or error.code not in _TRANSIENT_STATUSES
      ):
        raise
      delay = _backoff_delay(error, attempt)
      logging.info("HTTP %d for %s; retrying in %.0f s", error.code, url, delay)
      time.sleep(delay)


def ensure_annotations(
    mirror_dir: str, split: str, category: str, num_shards: int
) -> list[str]:
  """Mirror annotation shards 0..num_shards-1; returns their paths."""
  paths = []
  for shard in range(num_shards):
    path = layout.annotation_path(mirror_dir, split, category, shard)
    if not os.path.exists(path):
      name = f"{split}/annotations/{category}/annotations-{shard:05d}.jsonl.gz"
      logging.info("Fetching %s", name)
      _download(f"{_DATASET_BASE}/{name}", path)
    paths.append(path)
  return paths


def ensure_bench_annotations(mirror_dir: str) -> dict[str, str]:
  """Mirror all four bench horizon files; returns horizon -> path."""
  paths = {}
  for horizon in layout.BENCH_HORIZONS:
    path = layout.bench_annotation_path(mirror_dir, horizon)
    if not os.path.exists(path):
      logging.info("Fetching strideqa_bench_%s.json", horizon)
      _download(
          f"{_BENCH_BASE}/annotation_files/strideqa_bench_{horizon}.json",
          path,
      )
    paths[horizon] = path
  return paths


def ensure_bench_images(mirror_dir: str, images: set[str]) -> None:
  """Plain per-file downloads of bench images into the mirror."""
  missing = sorted(
      image
      for image in images
      if not os.path.exists(layout.bench_image_path(mirror_dir, image))
  )
  if not missing:
    logging.info("Mirror already holds all %d bench images", len(images))
    return
  logging.info(
      "Fetching %d of %d bench images (%d cached)",
      len(missing),
      len(images),
      len(images) - len(missing),
  )

  def fetch(image: str) -> None:
    _download(
        f"{_BENCH_BASE}/{image}", layout.bench_image_path(mirror_dir, image)
    )

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_DOWNLOAD_WORKERS.value
  ) as pool:
    for _ in pool.map(fetch, missing):
      pass


################################################################################
# MARK: Explosion
################################################################################


def explode_conversations(
    conversations: list[dict[str, str]],
) -> list[tuple[str, str]]:
  """Split a VILA conversation into its (human, gpt) turn pairs.

  Enforces strict alternation: turn 2i must be from "human" and turn 2i+1
  from "gpt". Raises ValueError on any violation — the invariant held over
  every source record inspected, so a violation means the source changed and
  the run must not silently emit misaligned pairs. An empty conversation
  returns no pairs.
  """
  if len(conversations) % 2:
    raise ValueError(
        f"conversation has {len(conversations)} turns; the trailing human"
        " turn has no gpt answer"
    )
  pairs = []
  for i in range(0, len(conversations), 2):
    human, gpt = conversations[i], conversations[i + 1]
    if human.get("from") != "human" or gpt.get("from") != "gpt":
      raise ValueError(
          f"turns ({i}, {i + 1}) are"
          f" ({human.get('from')!r}, {gpt.get('from')!r}), not"
          " ('human', 'gpt')"
      )
    pairs.append((human["value"], gpt["value"]))
  return pairs


def _record_pairs(record: dict) -> list[tuple[str, str]]:
  """A record's QA pairs, with the qa_info alignment invariant enforced.

  An empty conversation is exempt: it emits zero records either way, and the
  source treats it as valid (logged by the caller, not an error).
  """
  pairs = explode_conversations(record["conversations"])
  if pairs and len(record["qa_info"]) != len(pairs):
    raise ValueError(
        f"record {record['id']}: {len(pairs)} QA pairs but"
        f" {len(record['qa_info'])} qa_info entries"
    )
  return pairs


################################################################################
# MARK: Records
################################################################################


def make_tf_example(
    features_dict: dict[str, bytes | list[bytes]],
) -> tf.train.Example:
  """Create a tf.train.Example proto from a dictionary of byte features."""
  tf_features = {
      k: tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=v if isinstance(v, list) else [v]
          )
      )
      for k, v in features_dict.items()
  }
  return tf.train.Example(features=tf.train.Features(feature=tf_features))


def _json_bytes(value) -> bytes:
  """A field serialized as JSON; absent (None) becomes empty bytes."""
  if value is None:
    return b""
  return json.dumps(value, ensure_ascii=False).encode()


def record_filenames(record: dict) -> list[str]:
  """A dataset record's image filenames: 1 (spatial) or 4 (spatiotemporal,
  source order, current frame last)."""
  if "images" in record:
    return list(record["images"])
  return [record["image"]]


def build_dataset_features(
    record: dict, pair_index: int, images: list[bytes]
) -> dict[str, bytes | list[bytes]]:
  """The uniform feature set for one Dataset QA pair."""
  prompt, target = _record_pairs(record)[pair_index]
  qa_info = record["qa_info"][pair_index]
  return {
      "id": f"{record['id']}/{pair_index:02d}".encode(),
      "image/encoded": images,
      "image/filenames_json": _json_bytes(record_filenames(record)),
      "image/format": b"jpeg",
      "prompt": prompt.encode(),
      "target_text": target.encode(),
      "qa_category": qa_info["category"].encode(),
      "qa_info_json": _json_bytes(qa_info),
      "gt_value_json": b"",
      "unit_json": b"",
      "group_id": b"",
      "horizon": b"",
      "region_id": b"",
      "object_class": b"",
      "bbox_json": _json_bytes(record["bbox"]),
      "rle_json": _json_bytes(record["rle"]),
      # Spatial records carry `region`; spatiotemporal records omit the key
      # entirely. Either way an absent value becomes empty bytes.
      "region_json": _json_bytes(record.get("region")),
      "image_info_json": _json_bytes(record["image_info"]),
      "token_info_json": _json_bytes(record["token_info"]),
      "sample_id": record["id"].encode(),
      "pair_index": str(pair_index).encode(),
      "category": record["category"].encode(),
      "split": record["split"].encode(),
  }


def build_bench_features(
    record: dict, horizon: str, images: list[bytes]
) -> dict[str, bytes | list[bytes]]:
  """The uniform feature set for one Bench question."""
  return {
      "id": record["question_id"].encode(),
      "image/encoded": images,
      "image/filenames_json": _json_bytes(record["images"]),
      "image/format": b"jpeg",
      "prompt": record["question"].encode(),
      "target_text": record["gt"].encode(),
      "qa_category": record["qa_category"].encode(),
      "qa_info_json": b"",
      "gt_value_json": _json_bytes(record["gt_value"]),
      "unit_json": _json_bytes(record["unit"]),
      "group_id": record["group_id"].encode(),
      "horizon": horizon.encode(),
      "region_id": str(record["region_id"]).encode(),
      "object_class": record["object_class"].encode(),
      "bbox_json": b"",
      "rle_json": _json_bytes(record["rle"]),
      "region_json": b"",
      "image_info_json": b"",
      "token_info_json": _json_bytes(record["token_info"]),
      "sample_id": b"",
      "pair_index": b"",
      "category": b"",
      "split": b"",
  }


def write_bagz(
    examples, output_dir: str, stem: str, num_records: int
) -> int:
  """Write examples into `{stem}-NNNNN-of-NNNNN.bagz` shards; returns count."""
  num_shards = max(1, math.ceil(num_records / _RECORDS_PER_SHARD.value))
  os.makedirs(output_dir, exist_ok=True)
  written = 0
  shard_index = -1
  writer = None
  try:
    for example in examples:
      target_shard = min(written // _RECORDS_PER_SHARD.value, num_shards - 1)
      if target_shard != shard_index:
        if writer is not None:
          writer.close()
        shard_index = target_shard
        path = os.path.join(
            output_dir, f"{stem}-{shard_index:05d}-of-{num_shards:05d}.bagz"
        )
        logging.info("Writing %s", os.path.basename(path))
        # Default (auto) compression: JPEG payloads are incompressible, but
        # this keeps shards readable by a plain `bagz.Reader(path)`, which
        # `CompressionNone` does not.
        writer = bagz.Writer(path)
      # Deterministic serialization sorts the feature map. Without it,
      # protobuf emits map entries in an order that varies per process, so
      # two runs over identical input produce different bytes and the output
      # is not reproducible.
      writer.write(example.SerializeToString(deterministic=True))
      written += 1
  finally:
    if writer is not None:
      writer.close()
  if written != num_records:
    raise RuntimeError(f"expected {num_records} records but wrote {written}")
  return num_shards


################################################################################
# MARK: Dataset conversion
################################################################################


def _iter_annotation_records(paths: list[str]):
  for path in paths:
    with gzip.open(path, "rt") as handle:
      for line in handle:
        yield json.loads(line)


def convert_dataset(mirror_dir: str, output_dir: str) -> None:
  """One (split, category) combo -> one Bagz spec plus its sidecar."""
  split, category = _SPLIT.value, _CATEGORY.value
  available = layout.NUM_ANNOTATION_SHARDS[(split, category)]
  num_ann = _ANNOTATION_SHARDS.value
  if num_ann == -1:
    num_ann = available
  if not 1 <= num_ann <= available:
    raise app.UsageError(
        f"{split}/{category} has {available} annotation shards, not {num_ann}"
    )
  ann_paths = ensure_annotations(mirror_dir, split, category, num_ann)

  # Pass 1: count QA pairs per record, enforcing the explosion invariants.
  pair_counts = []
  num_empty = 0
  for record in _iter_annotation_records(ann_paths):
    count = len(_record_pairs(record))
    if count == 0:
      num_empty += 1
      logging.info("record %s has an empty conversation", record["id"])
    pair_counts.append(count)
  total_pairs = sum(pair_counts)
  logging.info(
      "%d records in %d annotation shard(s): %d QA pairs, %d empty",
      len(pair_counts),
      num_ann,
      total_pairs,
      num_empty,
  )

  # Uniform sample over QA-pair records, not source samples, so every pair in
  # the mirrored shards is equally likely regardless of conversation length.
  max_records = _MAX_RECORDS.value
  if 0 <= max_records < total_pairs:
    order = np.random.default_rng(_SEED.value).permutation(total_pairs)
    selected = {int(i) for i in order[:max_records]}
  else:
    selected = None  # everything
  num_selected = len(selected) if selected is not None else total_pairs
  logging.info("Selected %d of %d QA pairs", num_selected, total_pairs)

  # Pass 2: the images those pairs need.
  filenames: set[str] = set()
  base = 0
  for record, count in zip(_iter_annotation_records(ann_paths), pair_counts):
    if selected is None or any(base + i in selected for i in range(count)):
      filenames.update(record_filenames(record))
    base += count
  ensure_images(mirror_dir, split, filenames)

  # Pass 3: stream the selected pairs into Bagz shards.
  def examples():
    base = 0
    for record, count in zip(
        _iter_annotation_records(ann_paths), pair_counts
    ):
      wanted = [
          i
          for i in range(count)
          if selected is None or base + i in selected
      ]
      if wanted:
        images = []
        for name in record_filenames(record):
          with open(layout.image_path(mirror_dir, split, name), "rb") as f:
            images.append(f.read())
        for i in wanted:
          yield make_tf_example(build_dataset_features(record, i, images))
      base += count

  stem = f"strideqa_{split}_{category}"
  num_shards = write_bagz(examples(), output_dir, stem, num_selected)
  logging.info(
      "Wrote %d records across %d shard(s)", num_selected, num_shards
  )

  metadata = {
      "source": f"{_DATASET_BASE}/",
      "license": "CC BY-NC-SA 4.0",
      "split": split,
      "category": category,
      "annotation_shards_mirrored": num_ann,
      "annotation_shards_available": available,
      "seed": _SEED.value,
      "max_records": max_records,
      "num_records": num_selected,
      "num_shards": num_shards,
      "num_source_samples": len(pair_counts),
      "num_source_qa_pairs": total_pairs,
      "num_empty_conversations": num_empty,
      "bagz_spec": os.path.join(output_dir, f"{stem}@{num_shards}.bagz"),
      "feature_keys": list(_FEATURE_KEYS),
  }
  metadata_path = os.path.join(output_dir, f"{stem}.metadata.json")
  with open(metadata_path, "w") as handle:
    json.dump(metadata, handle, indent=2, ensure_ascii=False)
  logging.info("Wrote %s", metadata_path)


################################################################################
# MARK: Bench conversion
################################################################################


def convert_bench(mirror_dir: str, output_dir: str) -> None:
  """One run -> thirteen qa_category-partitioned specs plus one sidecar."""
  ann_paths = ensure_bench_annotations(mirror_dir)

  # group_id -> qa_category -> (horizon, record), in t0 file order.
  groups: dict[str, dict[str, tuple[str, dict]]] = {}
  order: list[str] = []
  num_read = 0
  for horizon in layout.BENCH_HORIZONS:
    with open(ann_paths[horizon]) as handle:
      for record in json.load(handle):
        group_id = record["group_id"]
        if group_id not in groups:
          groups[group_id] = {}
          order.append(group_id)
        groups[group_id][record["qa_category"]] = (horizon, record)
        num_read += 1

  expected = {category for _, category in _BENCH_PARTITIONS}
  for group_id in order:
    if set(groups[group_id]) != expected:
      raise RuntimeError(
          f"group {group_id} has categories {sorted(groups[group_id])},"
          " expected one record per partition"
      )
  # The key-set check above cannot see a duplicate qa_category within a group
  # (the second record would silently overwrite the first).
  if num_read != len(_BENCH_PARTITIONS) * len(groups):
    raise RuntimeError(
        f"read {num_read} bench records for {len(groups)} groups; expected"
        f" exactly {len(_BENCH_PARTITIONS)} per group"
    )

  # `--max_records` counts scene groups here: each group yields 13 records,
  # one per partition, so any sample covers all partitions and the composite
  # metrics (which join a group across horizons) stay computable.
  max_records = _MAX_RECORDS.value
  if 0 <= max_records < len(order):
    perm = np.random.default_rng(_SEED.value).permutation(len(order))
    order = [order[int(i)] for i in np.sort(perm[:max_records])]
  logging.info("Selected %d of %d scene groups", len(order), len(groups))

  images: set[str] = set()
  for group_id in order:
    for _, record in groups[group_id].values():
      images.update(record["images"])
  ensure_bench_images(mirror_dir, images)

  def examples(qa_category: str, horizon: str):
    for group_id in order:
      _, record = groups[group_id][qa_category]
      image_bytes = []
      for image in record["images"]:
        with open(layout.bench_image_path(mirror_dir, image), "rb") as f:
          image_bytes.append(f.read())
      yield make_tf_example(
          build_bench_features(record, horizon, image_bytes)
      )

  partitions = []
  for horizon, qa_category in _BENCH_PARTITIONS:
    stem = f"strideqa_bench_{qa_category}"
    num_shards = write_bagz(
        examples(qa_category, horizon), output_dir, stem, len(order)
    )
    partitions.append({
        "qa_category": qa_category,
        "horizon": horizon,
        "num_records": len(order),
        "num_shards": num_shards,
        "bagz_spec": os.path.join(output_dir, f"{stem}@{num_shards}.bagz"),
    })
  logging.info(
      "Wrote %d partitions x %d records", len(partitions), len(order)
  )

  metadata = {
      "source": f"{_BENCH_BASE}/",
      "license": "CC BY-NC-SA 4.0",
      "seed": _SEED.value,
      "max_records": max_records,
      "num_groups": len(order),
      "num_source_groups": len(groups),
      "num_records": len(order) * len(partitions),
      "partitions": partitions,
      "feature_keys": list(_FEATURE_KEYS),
  }
  metadata_path = os.path.join(output_dir, "strideqa_bench.metadata.json")
  with open(metadata_path, "w") as handle:
    json.dump(metadata, handle, indent=2, ensure_ascii=False)
  logging.info("Wrote %s", metadata_path)


################################################################################
# MARK: Main
################################################################################


def main(argv):
  """Convert the selected source into Bagz."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if _SOURCE.value == "dataset":
    convert_dataset(_MIRROR_DIR.value, _OUTPUT_DIR.value)
  else:
    convert_bench(_MIRROR_DIR.value, _OUTPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)
