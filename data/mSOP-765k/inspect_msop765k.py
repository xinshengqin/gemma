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

"""Visualize records from a converted mSOP-765k Bagz dataset.

Samples ``--k`` examples and renders them to a self-contained HTML page: the
decoded advertisement image next to every field.

Either end of the pipeline can be inspected:

* ``--stage output`` reads the Bagz produced by ``convert_msop765k.py``, either
  from ``--bagz`` directly or from ``--output_dir``/``--split`` via the metadata
  sidecar, and shows the stored features.
* ``--stage input`` reads the raw mirror — the parquet and the per-label image
  tarballs — and shows the untransformed source fields, so the two pages can be
  compared side by side to see what the conversion did.
"""

import base64
import html
import io
import json
import os
import tarfile
import textwrap

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from PIL import Image
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
_SPLIT = flags.DEFINE_enum("split", "test", ["test", "train"], "Split to read.")
_K = flags.DEFINE_integer("k", 5, "Number of examples to visualize.")
_SEED = flags.DEFINE_integer("seed", 0, "Seed for sampling.")
_HTML = flags.DEFINE_string("html", "", "Output HTML path.")

# Rendered before the remaining features, so the image-derived answer is easy to
# compare against the picture.
_LEAD_KEYS = ("id", "product_id", "filename", "image/format")

################################################################################
# MARK: Reading
################################################################################


def bagz_spec(output_dir: str, split: str) -> tuple[str, dict]:
  """Resolve the ``name@N.bagz`` spec for a converted split."""
  metadata_path = os.path.join(output_dir, f"msop765k_{split}.metadata.json")
  with open(metadata_path) as handle:
    metadata = json.load(handle)
  spec = os.path.join(
      output_dir, f"msop765k_{split}@{metadata['num_shards']}.bagz"
  )
  return spec, metadata


def parse_record(record: bytes) -> dict[str, bytes]:
  example = tf.train.Example()
  example.ParseFromString(record)
  return {
      key: value.bytes_list.value[0]
      for key, value in example.features.feature.items()
  }


################################################################################
# MARK: Rendering
################################################################################

_STYLE = """
:root{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;
  --sub:#5b6570;--line:#e6e9ee;--th:#8a94a0;--code:#eef1f6;--na:#b6bcc4}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#171b21;
  --ink:#e6e9ee;--sub:#9aa4b0;--line:#262c35;--th:#7a8593;--code:#1f2530;
  --na:#5a636e}}
:root[data-theme=dark]{--bg:#0e1116;--card:#171b21;--ink:#e6e9ee;--sub:#9aa4b0;
  --line:#262c35;--th:#7a8593;--code:#1f2530;--na:#5a636e}
:root[data-theme=light]{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--sub:#5b6570;
  --line:#e6e9ee;--th:#8a94a0;--code:#eef1f6;--na:#b6bcc4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,
  sans-serif}
header{max-width:1080px;margin:0 auto;padding:28px 24px 4px}
h1{margin:0 0 6px;font-size:21px;letter-spacing:-.01em}
.lede{margin:0;color:var(--sub)}
.lede code{background:var(--code);padding:1px 6px;border-radius:5px;
  font-size:.85em}
main{max-width:1080px;margin:0 auto;padding:16px 24px 48px;display:grid;
  gap:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;display:grid;grid-template-columns:300px 1fr}
@media(max-width:760px){.card{grid-template-columns:1fr}}
.imgwrap{background:#fff;padding:14px;display:flex;align-items:center;
  justify-content:center;border-right:1px solid var(--line)}
.imgwrap img{max-width:100%;height:auto;border-radius:6px}
.meta{padding:14px 16px;min-width:0}
.rid{font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--th);margin-bottom:10px}
.rid code{background:var(--code);color:var(--sub);padding:1px 6px;
  border-radius:5px;text-transform:none;letter-spacing:0;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:5px 0;vertical-align:top;
  border-bottom:1px solid var(--line)}
tr:last-child th,tr:last-child td{border-bottom:0}
th{color:var(--th);font-weight:500;width:150px;white-space:nowrap;
  padding-right:12px}
td{word-break:break-word}
.na{color:var(--na)}
pre{background:var(--code);border-radius:8px;padding:10px 12px;margin:10px 0 0;
  overflow-x:auto;font-size:12.5px;line-height:1.5}
.sec{margin-top:12px;font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--th)}
footer{max-width:1080px;margin:0 auto;padding:0 24px 40px;color:var(--sub);
  font-size:13px}
"""


def _row(key: str, value: str) -> str:
  shown = html.escape(value) if value else '<span class="na">∅ empty</span>'
  return f"<tr><th>{html.escape(key)}</th><td>{shown}</td></tr>"


def _card(
    index: int,
    ident: str,
    image_bytes: bytes,
    rows: list[tuple[str, str]],
    blocks: list[tuple[str, str]],
) -> str:
  """Render one example: image on the left, fields and text blocks right."""
  with Image.open(io.BytesIO(image_bytes)) as image:
    width, height = image.size
    fmt = image.format
  encoded = base64.b64encode(image_bytes).decode()
  table = "".join(_row(key, value) for key, value in rows)
  sections = "".join(
      f'<div class="sec">{html.escape(label)}</div>'
      f"<pre>{html.escape(text)}</pre>"
      for label, text in blocks
  )
  return f"""<article class="card">
  <div class="imgwrap">
    <img src="data:image/jpeg;base64,{encoded}" alt="advertisement">
  </div>
  <div class="meta">
    <div class="rid">example {index} &middot;
      <code>{html.escape(ident)}</code>
      &middot; {fmt} {width}&times;{height}px &middot;
      {len(image_bytes) / 1024:.0f} KB</div>
    <table>{table}</table>
    {sections}
  </div>
</article>"""


def render_output_card(index: int, features: dict[str, bytes]) -> str:
  """A converted record: every stored feature, verbatim."""
  image_bytes = features.get("image/encoded", b"")
  decoded = {
      key: value.decode("utf-8", errors="replace")
      for key, value in features.items()
      if key != "image/encoded"
  }
  target_json = decoded.pop("target_json", "")
  blocks = [
      ("prompt_system", decoded.pop("prompt_system", "")),
      ("prompt", decoded.pop("prompt", "")),
  ]
  try:
    blocks.append(
        ("target_json", json.dumps(json.loads(target_json), indent=2,
                                   ensure_ascii=False))
    )
  except json.JSONDecodeError:
    blocks.append(("target_json", target_json))

  ident = decoded.get("id", str(index))
  lead = [(k, decoded.pop(k)) for k in _LEAD_KEYS if k in decoded]
  rest = [(k, decoded[k]) for k in sorted(decoded)]
  return _card(index, ident, image_bytes, lead + rest, blocks)


def render_input_card(index: int, row, image_bytes: bytes) -> str:
  """A raw source row: parquet columns exactly as they are stored."""
  fields = []
  for column in row.index:
    value = row[column]
    fields.append((
        column,
        "" if pd.isna(value) else str(value),
    ))
  ident = f"{row['label']}/{row['filename']}"
  return _card(index, ident, image_bytes, fields, [])


################################################################################
# MARK: Main
################################################################################


def _sample(total: int, count: int) -> list[int]:
  """Pick indices without replacement; records are label-grouped on disk."""
  return sorted(
      int(i)
      for i in np.random.default_rng(_SEED.value).choice(
          total, min(count, total), replace=False
      )
  )


def _render_output(split: str) -> tuple[str, str, str]:
  """Returns (cards, source label, subtitle) for the converted Bagz."""
  if _BAGZ.value:
    spec, note = _BAGZ.value, ""
  else:
    spec, metadata = bagz_spec(_OUTPUT_DIR.value, split)
    note = (
        f" in {metadata['num_shards']} shard(s), drawn from"
        f" {metadata['source_records_in_split']} in the source split"
    )
  reader = bagz.Reader(spec)
  total = len(reader)
  logging.info("Opened %s: %d records", spec, total)

  cards = []
  for index in _sample(total, _K.value):
    features = parse_record(reader[index])
    cards.append(render_output_card(index, features))
    print(f"[{index}] {features.get('id', b'').decode()}")
  subtitle = (
      f"{len(cards)} of {total} record(s){note}. Every stored feature is shown"
      " verbatim."
  )
  return "".join(cards), os.path.basename(spec), subtitle


def _render_input(split: str) -> tuple[str, str, str]:
  """Returns (cards, source label, subtitle) for the raw mirror."""
  mirror = _MIRROR_DIR.value
  frame = pd.read_parquet(layout.parquet_path(mirror, split))
  frame = frame.astype({"label": str, "filename": str})
  total = len(frame)

  # The mirror is an incremental cache, so it usually holds shards for only
  # some labels. Sampling the whole parquet would keep landing on rows whose
  # image has not been fetched, so restrict to what is actually present.
  shard_dir = os.path.join(mirror, layout.IMAGE_DIR, split)
  mirrored = {
      name.removesuffix(".tar.gz")
      for name in os.listdir(shard_dir)
      if name.endswith(".tar.gz")
  }
  frame = frame[frame["label"].isin(mirrored)].reset_index(drop=True)
  if frame.empty:
    raise app.UsageError(
        f"no rows in {split}.parquet have a mirrored shard under {shard_dir}"
    )
  logging.info(
      "Read %s: %d of %d rows have a mirrored image",
      layout.parquet_path(mirror, split),
      len(frame),
      total,
  )

  cards = []
  for index in _sample(len(frame), _K.value):
    row = frame.iloc[index]
    shard = layout.shard_path(mirror, split, row["label"])
    with tarfile.open(shard, "r:gz") as tar:
      member = tar.extractfile(f"{row['label']}/{row['filename']}")
      image_bytes = member.read()
    cards.append(render_input_card(index, row, image_bytes))
    print(f"[{index}] {row['label']}/{row['filename']}")
  subtitle = (
      f"{len(cards)} of {len(frame)} mirrored row(s) &mdash; {total} in"
      f" <code>{split}.parquet</code> overall &mdash; each image read from its"
      " label tarball. Fields are the untransformed source values."
  )
  return "".join(cards), f"{os.path.basename(mirror)}/{split}.parquet", subtitle


def main(argv):
  """Sample k examples from one end of the pipeline and write an HTML page."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  split, stage = _SPLIT.value, _STAGE.value
  cards, source, subtitle = (
      _render_input(split) if stage == "input" else _render_output(split)
  )

  html_path = _HTML.value or f"msop765k_{split}_{stage}.html"
  os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
  document = textwrap.dedent(f"""\
      <title>mSOP-765k {split} &mdash; pipeline {stage}</title>
      <style>{_STYLE}</style>
      <header>
        <h1>mSOP-765k &mdash; pipeline {stage} &middot;
        <code>{html.escape(split)}</code></h1>
        <p class="lede">{subtitle} Sampled with seed {_SEED.value} from
        <code>{html.escape(source)}</code>.</p>
      </header>
      <main>{cards}</main>
      <footer>Source: retail-product-promotion/mSOP-765k
      (Lamm &amp; Keuper, TMLR 01/2026).</footer>
      """)
  with open(html_path, "w") as handle:
    handle.write(document)
  logging.info("Wrote %s", html_path)
  print(f"\nwrote {html_path}")


if __name__ == "__main__":
  app.run(main)
