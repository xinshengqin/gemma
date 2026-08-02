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

"""Visualize records from either end of the STRIDE-QA pipeline.

Samples ``--k`` examples and renders them to a self-contained HTML page. Each
frame is shown as a raw + overlay pair: bounding boxes as rectangles and RLE
masks as outlines, labeled ``Region [k]`` — so the regions the text refers to
can be checked against the pixels. Multi-frame records render as a filmstrip
with the current (last) frame emphasized.

Display copies are downscaled to ~960 px on the longest edge; the original
dimensions are printed on every card. This deviates from the mSOP-765k
viewer's verbatim embedding: at 2880x1860 x up to 8 renders per record, a
verbatim page would run to hundreds of megabytes.

Either end of the pipeline can be inspected:

* ``--stage output`` reads the converted Bagz, resolved from the metadata
  sidecar in ``--output_dir`` (or ``--bagz`` directly), and shows the stored
  features.
* ``--stage input`` reads the raw mirror and shows the untransformed source
  records — the full conversation for the Dataset, the question and ground
  truth for Bench — so the two pages can be compared side by side.
"""

import base64
import gzip
import html
import io
import json
import os
import textwrap

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pycocotools import mask as mask_utils
import tensorflow as tf

import bagz

import layout

################################################################################
# MARK: Flags
################################################################################

_STAGE = flags.DEFINE_enum(
    "stage",
    "output",
    ["input", "output"],
    "Which end of the pipeline to render.",
)
_SOURCE = flags.DEFINE_enum(
    "source", "dataset", ["dataset", "bench"], "Source repository to read."
)
_BAGZ = flags.DEFINE_string(
    "bagz",
    "",
    "stage=output: path or `name@N.bagz` spec to read. Overrides --output_dir.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "out", "stage=output: directory holding the converted shards."
)
_MIRROR_DIR = flags.DEFINE_string(
    "mirror_dir", "mirror", "stage=input: directory holding the raw mirror."
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
_K = flags.DEFINE_integer("k", 5, "Number of examples to visualize.")
_SEED = flags.DEFINE_integer("seed", 0, "Seed for sampling.")
_HTML = flags.DEFINE_string("html", "", "Output HTML path.")

_DISPLAY_EDGE = 960  # longest edge of a display copy, px

# Frame order of 4-frame records, oldest first, current last — the order the
# official bench inference code zips `images` with.
_TIME_BUCKETS = ("prev_3", "prev_2", "prev_1", "current")

_PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#469990", "#9a6324", "#800000", "#000075",
)

################################################################################
# MARK: Overlays
################################################################################


def _color(index: int) -> str:
  return _PALETTE[index % len(_PALETTE)]


def _font(size: int):
  try:
    return ImageFont.load_default(size=size)
  except TypeError:  # older Pillow: fixed-size bitmap font only
    return ImageFont.load_default()


def _mask_outline(rle: dict, size: tuple[int, int]) -> np.ndarray:
  """A boolean outline of one RLE mask, resized to the display size."""
  rle = dict(rle)
  if isinstance(rle["counts"], str):
    rle["counts"] = rle["counts"].encode()
  decoded = mask_utils.decode(rle)
  mask = np.asarray(
      Image.fromarray(decoded * 255).resize(size, Image.NEAREST)
  ).astype(bool)
  core = mask
  for axis in (0, 1):
    for shift in (1, -1):
      core = core & np.roll(mask, shift, axis)
  edge = mask & ~core
  # Thicken to ~3 px so the outline survives JPEG re-encoding.
  thick = edge.copy()
  for axis in (0, 1):
    for shift in (1, -1):
      thick |= np.roll(edge, shift, axis)
  return thick


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
  return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


class _FrameOverlay:
  """Boxes and masks to draw on one frame, all in source coordinates."""

  def __init__(self):
    self.boxes: list[tuple[list[float], int]] = []  # (x1 y1 x2 y2, region k)
    self.masks: list[tuple[dict, int]] = []  # (rle, region k)


def _overlays_for_frames(
    num_frames: int,
    bbox_data,
    rle_data,
    region_id: int | None,
) -> list[_FrameOverlay]:
  """Distribute a record's bbox/rle annotations over its frames.

  Spatial records annotate their single frame with flat lists (index k is
  region k). Spatiotemporal and bench records hold per-frame dicts keyed
  ``prev_3 .. current``; bench frames carry a single mask, labeled with the
  record's ``region_id``. Dataset prev-frame entries are placeholders — an
  empty bbox and an RLE stripped to its tokens — so anything without real
  geometry is skipped rather than drawn.
  """
  overlays = [_FrameOverlay() for _ in range(num_frames)]

  def add(overlay: _FrameOverlay, boxes, rles):
    if isinstance(rles, dict):  # bench: one mask per frame
      if "counts" in rles:
        overlay.masks.append((rles, region_id or 0))
    else:
      for k, rle in enumerate(rles or []):
        if isinstance(rle, dict) and "counts" in rle and "size" in rle:
          overlay.masks.append((rle, k))
    for k, box in enumerate(boxes or []):
      if isinstance(box, (list, tuple)) and len(box) == 4:
        overlay.boxes.append((box, k))

  if isinstance(bbox_data, dict) or isinstance(rle_data, dict):
    for index, overlay in enumerate(overlays):
      bucket = _TIME_BUCKETS[index] if num_frames == 4 else "current"
      boxes = (bbox_data or {}).get(bucket)
      rles = (rle_data or {}).get(bucket)
      add(overlay, boxes, rles)
  elif num_frames:
    add(overlays[0], bbox_data, rle_data)
  return overlays


def _render_frame(
    image_bytes: bytes, overlay: _FrameOverlay | None
) -> tuple[str, tuple[int, int]]:
  """One display copy as a base64 JPEG; returns (data URI, original size)."""
  with Image.open(io.BytesIO(image_bytes)) as image:
    original = image.size
    scale = _DISPLAY_EDGE / max(original)
    display_size = (
        round(original[0] * scale),
        round(original[1] * scale),
    )
    display = image.convert("RGB").resize(display_size, Image.LANCZOS)

  if overlay is not None:
    draw = ImageDraw.Draw(display)
    font = _font(15)
    pixels = None
    for rle, k in overlay.masks:
      outline = _mask_outline(rle, display_size)
      if pixels is None:
        pixels = np.asarray(display)
      pixels = pixels.copy()
      pixels[outline] = _hex_to_rgb(_color(k))
      display = Image.fromarray(pixels)
      draw = ImageDraw.Draw(display)
      rows, cols = np.nonzero(outline)
      if rows.size:
        draw.text(
            (int(cols.min()), max(0, int(rows.min()) - 18)),
            f"Region [{k}]",
            fill=_color(k),
            font=font,
        )
    for box, k in overlay.boxes:
      x1, y1, x2, y2 = (value * scale for value in box)
      draw.rectangle((x1, y1, x2, y2), outline=_color(k), width=2)
      draw.text(
          (x1 + 3, max(0, y1 - 18)),
          f"Region [{k}]",
          fill=_color(k),
          font=font,
      )

  buffer = io.BytesIO()
  display.save(buffer, "JPEG", quality=85)
  encoded = base64.b64encode(buffer.getvalue()).decode()
  return f"data:image/jpeg;base64,{encoded}", original


################################################################################
# MARK: HTML
################################################################################

_STYLE = """
:root{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;
  --sub:#5b6570;--line:#e6e9ee;--th:#8a94a0;--code:#eef1f6;--na:#b6bcc4;
  --accent:#3f6ea8}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#171b21;
  --ink:#e6e9ee;--sub:#9aa4b0;--line:#262c35;--th:#7a8593;--code:#1f2530;
  --na:#5a636e;--accent:#6ba3e8}}
:root[data-theme=dark]{--bg:#0e1116;--card:#171b21;--ink:#e6e9ee;--sub:#9aa4b0;
  --line:#262c35;--th:#7a8593;--code:#1f2530;--na:#5a636e;--accent:#6ba3e8}
:root[data-theme=light]{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--sub:#5b6570;
  --line:#e6e9ee;--th:#8a94a0;--code:#eef1f6;--na:#b6bcc4;--accent:#3f6ea8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,
  sans-serif}
header{max-width:1280px;margin:0 auto;padding:28px 24px 4px}
h1{margin:0 0 6px;font-size:21px;letter-spacing:-.01em}
.lede{margin:0;color:var(--sub)}
.lede code{background:var(--code);padding:1px 6px;border-radius:5px;
  font-size:.85em}
main{max-width:1280px;margin:0 auto;padding:16px 24px 48px;display:grid;
  gap:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;padding:14px 16px}
.rid{font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--th);margin-bottom:10px}
.rid code{background:var(--code);color:var(--sub);padding:1px 6px;
  border-radius:5px;text-transform:none;letter-spacing:0;font-size:12px}
.strip{display:flex;gap:12px;overflow-x:auto;padding-bottom:6px}
.frame{flex:0 0 auto;width:300px}
.frame.current{width:340px}
.frame .tag{font-size:11px;color:var(--th);text-transform:uppercase;
  letter-spacing:.06em;margin:2px 0 4px}
.frame.current .tag{color:var(--accent);font-weight:600}
.frame img{width:100%;height:auto;border-radius:6px;display:block;
  margin-bottom:6px}
.frame.current img{outline:2px solid var(--accent);outline-offset:-2px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:10px}
th,td{text-align:left;padding:5px 0;vertical-align:top;
  border-bottom:1px solid var(--line)}
tr:last-child th,tr:last-child td{border-bottom:0}
th{color:var(--th);font-weight:500;width:170px;white-space:nowrap;
  padding-right:12px}
td{word-break:break-word}
.na{color:var(--na)}
pre{background:var(--code);border-radius:8px;padding:10px 12px;margin:10px 0 0;
  overflow-x:auto;font-size:12.5px;line-height:1.5;white-space:pre-wrap}
.sec{margin-top:12px;font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--th)}
.turn{margin:6px 0}
.turn b{color:var(--accent)}
footer{max-width:1280px;margin:0 auto;padding:0 24px 40px;color:var(--sub);
  font-size:13px}
"""


def _row(key: str, value: str) -> str:
  shown = html.escape(value) if value else '<span class="na">&empty; empty</span>'
  return f"<tr><th>{html.escape(key)}</th><td>{shown}</td></tr>"


def _pretty_json(text: str) -> str:
  try:
    return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
  except (json.JSONDecodeError, TypeError):
    return text


def _filmstrip(
    frames: list[bytes],
    overlays: list[_FrameOverlay],
    labels: list[str],
) -> tuple[str, str]:
  """Raw + overlay pairs for every frame; returns (html, dimensions note)."""
  cells = []
  original = None
  for index, (image_bytes, overlay) in enumerate(zip(frames, overlays)):
    current = index == len(frames) - 1
    css = "frame current" if current and len(frames) > 1 else "frame"
    raw_uri, original = _render_frame(image_bytes, None)
    overlay_uri, _ = _render_frame(image_bytes, overlay)
    label = html.escape(labels[index]) if index < len(labels) else ""
    tag = f"{label}" + (" &middot; current" if current and len(frames) > 1 else "")
    cells.append(
        f'<div class="{css}"><div class="tag">{tag}</div>'
        f'<img src="{raw_uri}" alt="raw">'
        f'<img src="{overlay_uri}" alt="overlay"></div>'
    )
  note = (
      f"{original[0]}&times;{original[1]}px original, shown at"
      f" {_DISPLAY_EDGE}px; raw above, regions overlaid below"
  )
  return f'<div class="strip">{"".join(cells)}</div>', note


def _card(
    index: int,
    ident: str,
    strip_html: str,
    note: str,
    rows: list[tuple[str, str]],
    blocks: list[tuple[str, str]],
) -> str:
  table = "".join(_row(key, value) for key, value in rows)
  sections = "".join(
      f'<div class="sec">{html.escape(label)}</div><pre>{body}</pre>'
      for label, body in blocks
  )
  return f"""<article class="card">
  <div class="rid">example {index} &middot; <code>{html.escape(ident)}</code>
    &middot; {note}</div>
  {strip_html}
  <table>{table}</table>
  {sections}
</article>"""


################################################################################
# MARK: Output stage
################################################################################


def parse_record(record: bytes) -> dict[str, list[bytes]]:
  example = tf.train.Example()
  example.ParseFromString(record)
  return {
      key: list(value.bytes_list.value)
      for key, value in example.features.feature.items()
  }


def _first(features: dict[str, list[bytes]], key: str) -> str:
  values = features.get(key, [b""])
  return values[0].decode("utf-8", errors="replace") if values else ""


def _load_json_feature(features: dict[str, list[bytes]], key: str):
  text = _first(features, key)
  return json.loads(text) if text else None


def render_output_card(index: int, features: dict[str, list[bytes]]) -> str:
  """A converted record: every stored feature, the frames with overlays."""
  frames = features["image/encoded"]
  filenames = _load_json_feature(features, "image/filenames_json") or []
  region_id = _first(features, "region_id")
  overlays = _overlays_for_frames(
      len(frames),
      _load_json_feature(features, "bbox_json"),
      _load_json_feature(features, "rle_json"),
      int(region_id) if region_id else None,
  )
  labels = [os.path.basename(name) for name in filenames]
  strip, note = _filmstrip(frames, overlays, labels)

  decoded = {
      key: _first(features, key)
      for key in sorted(features)
      if key != "image/encoded"
  }
  blocks = [
      ("prompt", html.escape(decoded.pop("prompt", ""))),
      ("target_text", html.escape(decoded.pop("target_text", ""))),
  ]
  for key in ("qa_info_json", "gt_value_json", "unit_json"):
    value = decoded.pop(key, "")
    if value:
      blocks.append((key, html.escape(_pretty_json(value))))
  # The remaining *_json features stay in the table, truncated: bbox/rle
  # payloads run to hundreds of KB of coordinates.
  rows = []
  for key, value in decoded.items():
    if len(value) > 300:
      value = f"{value[:300]}… (+{len(value) - 300} chars)"
    rows.append((key, value))
  ident = _first(features, "id")
  return _card(index, ident, strip, note, rows, blocks)


def _read_output_specs() -> list[tuple[str, bagz.Reader]]:
  """The Bagz readers to sample from, resolved like production readers do."""
  if _BAGZ.value:
    return [(os.path.basename(_BAGZ.value), bagz.Reader(_BAGZ.value))]
  output_dir = _OUTPUT_DIR.value
  if _SOURCE.value == "dataset":
    stem = f"strideqa_{_SPLIT.value}_{_CATEGORY.value}"
    with open(os.path.join(output_dir, f"{stem}.metadata.json")) as handle:
      metadata = json.load(handle)
    spec = os.path.join(
        output_dir, f"{stem}@{metadata['num_shards']}.bagz"
    )
    return [(os.path.basename(spec), bagz.Reader(spec))]
  with open(
      os.path.join(output_dir, "strideqa_bench.metadata.json")
  ) as handle:
    metadata = json.load(handle)
  specs = []
  for partition in metadata["partitions"]:
    spec = os.path.join(
        output_dir, os.path.basename(partition["bagz_spec"])
    )
    specs.append((os.path.basename(spec), bagz.Reader(spec)))
  return specs


def _render_output() -> tuple[str, str, str]:
  """Returns (cards, source label, subtitle) for the converted Bagz."""
  specs = _read_output_specs()
  total = sum(len(reader) for _, reader in specs)
  logging.info(
      "Opened %d spec(s): %d records", len(specs), total
  )
  cards = []
  for index in _sample(total, _K.value):
    offset = index
    for name, reader in specs:
      if offset < len(reader):
        break
      offset -= len(reader)
    features = parse_record(reader[offset])
    cards.append(render_output_card(index, features))
    print(f"[{index}] {name}: {_first(features, 'id')}")
  label = (
      specs[0][0]
      if len(specs) == 1
      else f"{len(specs)} bench partition specs"
  )
  subtitle = (
      f"{len(cards)} of {total} record(s). Features are shown verbatim;"
      " long JSON payloads truncated for display."
  )
  return "".join(cards), label, subtitle


################################################################################
# MARK: Input stage
################################################################################


def _conversation_html(conversations: list[dict[str, str]]) -> str:
  turns = []
  for turn in conversations:
    role = html.escape(turn.get("from", "?"))
    value = html.escape(turn.get("value", ""))
    turns.append(f'<div class="turn"><b>{role}</b>: {value}</div>')
  if not turns:
    return '<span class="na">&empty; empty conversation</span>'
  return "".join(turns)


def render_dataset_input_card(index: int, record: dict, frames) -> str:
  """A raw Dataset sample: full conversation, sample-level annotations."""
  overlays = _overlays_for_frames(
      len(frames), record.get("bbox"), record.get("rle"), None
  )
  filenames = [record["image"]] if "image" in record else record["images"]
  strip, note = _filmstrip(frames, overlays, filenames)
  region = record.get("region")
  rows = [
      ("id", record["id"]),
      ("split", record["split"]),
      ("category", record["category"]),
      ("turns", str(len(record["conversations"]))),
      ("qa_info entries", str(len(record["qa_info"]))),
      ("qa categories", ", ".join(
          sorted({info["category"] for info in record["qa_info"]})
      )),
      ("image_info", json.dumps(record["image_info"], ensure_ascii=False)),
      ("token_info", json.dumps(record["token_info"], ensure_ascii=False)[:300]),
      # `region: null` must render as an absent value, not the string "null".
      ("region", "" if region is None else json.dumps(region)[:300]),
  ]
  blocks = [("conversations", _conversation_html(record["conversations"]))]
  return _card(index, record["id"], strip, note, rows, blocks)


def render_bench_input_card(index: int, record: dict, frames) -> str:
  """A raw Bench question: question, gt, and the per-frame mask."""
  overlays = _overlays_for_frames(
      len(frames), None, record.get("rle"), record.get("region_id")
  )
  labels = [os.path.basename(name) for name in record["images"]]
  strip, note = _filmstrip(frames, overlays, labels)
  rows = [
      ("question_id", record["question_id"]),
      ("group_id", record["group_id"]),
      ("qa_category", record["qa_category"]),
      ("object_class", record.get("object_class", "")),
      ("region_id", str(record.get("region_id", ""))),
      ("gt_value", json.dumps(record.get("gt_value"), ensure_ascii=False)),
      ("unit", json.dumps(record.get("unit"), ensure_ascii=False)),
  ]
  blocks = [
      ("question", html.escape(record["question"])),
      ("gt", html.escape(record["gt"])),
  ]
  return _card(index, record["question_id"], strip, note, rows, blocks)


def _mirrored_dataset_records() -> list[dict]:
  """Raw records in mirrored annotation shards whose images are all present."""
  mirror, split, category = (
      _MIRROR_DIR.value,
      _SPLIT.value,
      _CATEGORY.value,
  )
  records = []
  skipped = 0
  for shard in range(layout.NUM_ANNOTATION_SHARDS[(split, category)]):
    path = layout.annotation_path(mirror, split, category, shard)
    if not os.path.exists(path):
      continue
    with gzip.open(path, "rt") as handle:
      for line in handle:
        record = json.loads(line)
        filenames = (
            [record["image"]] if "image" in record else record["images"]
        )
        if all(
            os.path.exists(layout.image_path(mirror, split, name))
            for name in filenames
        ):
          records.append(record)
        else:
          skipped += 1
  if not records:
    raise app.UsageError(
        f"no {split}/{category} records have all images mirrored under"
        f" {mirror}; run the converter first"
    )
  logging.info(
      "%d mirrored records (%d lack images)", len(records), skipped
  )
  return records


def _mirrored_bench_records() -> list[dict]:
  mirror = _MIRROR_DIR.value
  records = []
  skipped = 0
  for horizon in layout.BENCH_HORIZONS:
    path = layout.bench_annotation_path(mirror, horizon)
    if not os.path.exists(path):
      continue
    with open(path) as handle:
      for record in json.load(handle):
        if all(
            os.path.exists(layout.bench_image_path(mirror, image))
            for image in record["images"]
        ):
          records.append(record)
        else:
          skipped += 1
  if not records:
    raise app.UsageError(
        f"no bench records have all images mirrored under {mirror}; run the"
        " converter first"
    )
  logging.info(
      "%d mirrored bench records (%d lack images)", len(records), skipped
  )
  return records


def _render_input() -> tuple[str, str, str]:
  """Returns (cards, source label, subtitle) for the raw mirror."""
  mirror = _MIRROR_DIR.value
  if _SOURCE.value == "dataset":
    records = _mirrored_dataset_records()
    split = _SPLIT.value

    def frames_of(record):
      filenames = (
          [record["image"]] if "image" in record else record["images"]
      )
      frames = []
      for name in filenames:
        with open(layout.image_path(mirror, split, name), "rb") as handle:
          frames.append(handle.read())
      return frames

    render = render_dataset_input_card
    label = f"{_SPLIT.value}/{_CATEGORY.value}"
  else:
    records = _mirrored_bench_records()

    def frames_of(record):
      frames = []
      for image in record["images"]:
        with open(layout.bench_image_path(mirror, image), "rb") as handle:
          frames.append(handle.read())
      return frames

    render = render_bench_input_card
    label = "bench annotation_files"

  cards = []
  for index in _sample(len(records), _K.value):
    record = records[index]
    cards.append(render(index, record, frames_of(record)))
    print(f"[{index}] {record.get('id') or record.get('question_id')}")
  subtitle = (
      f"{len(cards)} of {len(records)} mirrored record(s). Fields are the"
      " untransformed source values; images read from the mirror."
  )
  return "".join(cards), label, subtitle


################################################################################
# MARK: Main
################################################################################


def _sample(total: int, count: int) -> list[int]:
  """Pick indices without replacement, in reading order."""
  return sorted(
      int(i)
      for i in np.random.default_rng(_SEED.value).choice(
          total, min(count, total), replace=False
      )
  )


def main(argv):
  """Sample k examples from one end of the pipeline and write an HTML page."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  stage, source = _STAGE.value, _SOURCE.value
  cards, label, subtitle = (
      _render_input() if stage == "input" else _render_output()
  )

  html_path = _HTML.value or f"strideqa_{source}_{stage}.html"
  os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
  document = textwrap.dedent(f"""\
      <title>STRIDE-QA {source} &mdash; pipeline {stage}</title>
      <style>{_STYLE}</style>
      <header>
        <h1>STRIDE-QA &mdash; pipeline {stage} &middot;
        <code>{html.escape(source)}</code></h1>
        <p class="lede">{subtitle} Sampled with seed {_SEED.value} from
        <code>{html.escape(label)}</code>.</p>
      </header>
      <main>{cards}</main>
      <footer>Source: turing-motors/STRIDE-QA (arXiv 2508.10427),
      CC BY-NC-SA 4.0.</footer>
      """)
  with open(html_path, "w") as handle:
    handle.write(document)
  logging.info("Wrote %s", html_path)
  print(f"\nwrote {html_path}")


if __name__ == "__main__":
  app.run(main)
