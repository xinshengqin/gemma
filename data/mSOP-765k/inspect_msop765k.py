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

Reads the sharded Bagz collection produced by ``convert_msop765k.py``, samples
``--k`` records and renders them to a self-contained HTML page: the decoded
advertisement image next to every stored feature.
"""

import base64
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
import tensorflow as tf

import bagz

################################################################################
# MARK: Flags
################################################################################

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "out", "Directory holding the converted Bagz shards."
)
_SPLIT = flags.DEFINE_enum("split", "test", ["test", "train"], "Split to read.")
_K = flags.DEFINE_integer("k", 5, "Number of records to visualize.")
_SEED = flags.DEFINE_integer("seed", 0, "Seed for record sampling.")
_HTML = flags.DEFINE_string(
    "html", "", "Output HTML path. Defaults to <output_dir>/<split>_samples.html"
)

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


def render_card(index: int, features: dict[str, bytes]) -> str:
  """Render one record: image on the left, every stored feature on the right."""
  image_bytes = features.get("image/encoded", b"")
  with Image.open(io.BytesIO(image_bytes)) as image:
    width, height = image.size
    mode = image.format
  encoded = base64.b64encode(image_bytes).decode()

  decoded = {
      key: value.decode("utf-8", errors="replace")
      for key, value in features.items()
      if key != "image/encoded"
  }
  target_json = decoded.pop("target_json", "")
  prompt_system = decoded.pop("prompt_system", "")
  prompt = decoded.pop("prompt", "")

  lead = [_row(k, decoded.pop(k, "")) for k in _LEAD_KEYS if k in features]
  rest = [_row(k, decoded[k]) for k in sorted(decoded)]

  try:
    pretty = json.dumps(json.loads(target_json), indent=2, ensure_ascii=False)
  except json.JSONDecodeError:
    pretty = target_json

  return f"""<article class="card">
  <div class="imgwrap">
    <img src="data:image/jpeg;base64,{encoded}" alt="advertisement">
  </div>
  <div class="meta">
    <div class="rid">record {index} &middot;
      <code>{html.escape(decoded.get('id', str(index)))}</code>
      &middot; {mode} {width}&times;{height}px &middot;
      {len(image_bytes) / 1024:.0f} KB</div>
    <table>{''.join(lead)}{''.join(rest)}</table>
    <div class="sec">prompt_system</div><pre>{html.escape(prompt_system)}</pre>
    <div class="sec">prompt</div><pre>{html.escape(prompt)}</pre>
    <div class="sec">target_json</div><pre>{html.escape(pretty)}</pre>
  </div>
</article>"""


################################################################################
# MARK: Main
################################################################################


def main(argv):
  """Sample k records from a converted split and write an HTML preview."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  split = _SPLIT.value
  spec, metadata = bagz_spec(_OUTPUT_DIR.value, split)
  reader = bagz.Reader(spec)
  total = len(reader)
  logging.info("Opened %s: %d records", spec, total)

  count = min(_K.value, total)
  # Records are grouped by label on disk, so sample rather than take a prefix.
  indices = sorted(
      np.random.default_rng(_SEED.value).choice(total, count, replace=False)
  )

  cards = []
  for index in indices:
    features = parse_record(reader[int(index)])
    cards.append(render_card(int(index), features))
    summary = {
        key: features[key].decode("utf-8", errors="replace")
        for key in ("id", "brand", "price", "different_types")
        if key in features
    }
    print(f"[{index}] {summary}")

  html_path = _HTML.value or os.path.join(
      _OUTPUT_DIR.value, f"msop765k_{split}_samples.html"
  )
  document = textwrap.dedent(f"""\
      <title>mSOP-765k {split} &mdash; {count} sampled records</title>
      <style>{_STYLE}</style>
      <header>
        <h1>mSOP-765k &mdash; <code>{split}</code> split</h1>
        <p class="lede">{count} record(s) sampled with seed
        {_SEED.value} from <code>{html.escape(os.path.basename(spec))}</code>
        &mdash; {total} record(s) in {metadata['num_shards']} shard(s),
        drawn from {metadata['source_records_in_split']} in the source split.
        Images are 512px; every stored feature is shown verbatim.</p>
      </header>
      <main>{''.join(cards)}</main>
      <footer>Source: retail-product-promotion/mSOP-765k (TMLR 01/2026),
      CC BY-NC-ND 4.0.</footer>
      """)
  with open(html_path, "w") as handle:
    handle.write(document)
  logging.info("Wrote %s", html_path)
  print(f"\nwrote {html_path}")


if __name__ == "__main__":
  app.run(main)
