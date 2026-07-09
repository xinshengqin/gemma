"""Visual-Sudoku Bagz pipeline for DiffusionGemma SFT (design §4).

``make_sudoku_vision_ds`` replaces the text pipeline of
``gemma/diffusion/.../data/sudoku/sudoku_data.py``: the puzzle arrives as
**one image** of the 9x9 grid; the solved grid is still generated as text.
The prompt template carries a literal ``<|image|>`` placeholder instead of
the puzzle digits; two host-side (NumPy, pre-jit) transforms produce the
patch tensors and expand the placeholder; everything from ``CanvasChunker``
on is the baseline pipeline.

Bagz record layout (written by ``convert_sudoku_vision.py``):
  - ``puzzle_image``: PNG bytes of the rendered 9x9 puzzle grid.
  - ``solution``: space-separated solved grid string.
  - ``puzzle``: space-separated puzzle string (kept so the baseline Sudoku
    eval metrics, which need ``batch.puzzle_tokens`` to identify the masked
    cells, keep working unchanged).
"""

import dataclasses
import io
from typing import Any

from PIL import Image
import grain.python as grain
import numpy as np
import tensorflow as tf

from ft_gemma.diffusion.hackable_diffusion_adapter.data import vision_data
from gemma import gm
from gemma.diffusion.hackable_diffusion_adapter.data import data as adapter_data
from gemma.diffusion.hackable_diffusion_adapter.data.sudoku import sudoku_data
from kauldron import kd


@dataclasses.dataclass
class ParseSudokuVisionExample(grain.MapTransform):
  """Parses raw Bagz bytes (tf.train.Example) into image, prompt and solution.

  Decodes the puzzle image and emits the user-turn text with a literal
  ``<|image|>`` placeholder — the puzzle digits live only in the image.
  """

  def map(self, record_bytes: bytes) -> dict[str, Any]:
    example = tf.train.Example()
    example.ParseFromString(record_bytes)
    features = example.features.feature

    image_bytes = features["puzzle_image"].bytes_list.value[0]
    solution = features["solution"].bytes_list.value[0].decode("utf-8")
    puzzle = features["puzzle"].bytes_list.value[0].decode("utf-8")

    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    return {
        "prompt": "<|image|>",
        "image": image,
        "response": solution,
        "puzzle_raw": puzzle,
    }


_DEFAULT_SUDOKU_VISION_PROMPT = (
    "<|turn>system Solve the Sudoku puzzle shown in the image. Empty cells"
    " are represented by 0. Output ONLY the solved puzzle immediately as"
    " a 9x9 grid of numbers separated by spaces. Do not include ####,"
    " explanations, or any other text.<turn|>\n<|turn>user"
    " {text}<turn|>\n<|turn>model\n"
)


def make_sudoku_vision_ds(
    bagz_path: str,
    training: bool,
    batch_size: int,
    prompt_len: int,
    num_canvases: int,
    canvas_size: int,
    slice_start: int = 0,
    slice_stop: int | None = None,
    prompt_template: str = _DEFAULT_SUDOKU_VISION_PROMPT,
    max_soft_tokens_per_image: int = 280,
    num_workers: int = 16,
) -> sudoku_data.Bagz:
  """Build the visual-Sudoku dataset pipeline.

  Args:
    bagz_path: Path to the Bagz dataset (visual variant, see module doc).
    training: Whether this is training or evaluation.
    batch_size: Per-device batch size.
    prompt_len: Maximum prompt token length P — must hold the expanded image
      span (S_v + 4 <= max_soft_tokens_per_image + 4) plus the instruction
      text.
    num_canvases: Number of canvas chunks for the response.
    canvas_size: Token length of each canvas chunk.
    slice_start: Start index for dataset slicing.
    slice_stop: Stop index for dataset slicing.
    prompt_template: Chat template for the prompt. Must contain ``{text}``,
      which is replaced with the literal ``<|image|>`` placeholder.
    max_soft_tokens_per_image: Preprocessing budget S_max — must equal the
      model's ``vision_encoder.output_length``.
    num_workers: Number of workers for data loading.

  Returns:
    A ``Bagz`` dataset config.
  """
  tokenizer = gm.text.Gemma4Tokenizer()
  pad_token = gm.text.Gemma4Tokenizer.special_tokens.PAD
  eos_token = gm.text.Gemma4Tokenizer.special_tokens.EOS

  transforms = [
      # 1. Parse raw bytes: decode image, prompt text contains <|image|>.
      ParseSudokuVisionExample(),
  ]

  if not training:
    # (eval only) Copy raw response string BEFORE formatting CoT
    transforms.append(
        adapter_data.CopyField(src_key="response", dst_key="solution_raw")
    )

  transforms.extend([
      # 2. Format Sudoku Response
      gm.data.FormatText(
          key="response",
          template="{text}",
      ),
      # 3. Format Prompt (the {text} slot receives the <|image|> placeholder)
      gm.data.FormatText(
          key="prompt",
          template=prompt_template,
      ),
      # 4. Tokenize (<|image|> is a single reserved id, 258880)
      gm.data.Tokenize(tokenizer=tokenizer, key="prompt", add_bos=True),
      gm.data.Tokenize(tokenizer=tokenizer, key="response"),
      # 5. Preprocess image: resize (multiples of 48px, <= S_max*9 patches)
      #    + normalize [0,1] + patchify + pad. Also computes S_v.
      vision_data.PreprocessAndPatchifyImage(
          image_key="image",
          max_soft_tokens=max_soft_tokens_per_image,
      ),
      # 6. Expand <|image|> -> \n\n <soi> [-2] x S_v <eoi> \n\n
      vision_data.ExpandImagePlaceholders(
          key="prompt",
          soft_token_count_key="soft_token_count",
      ),
  ])

  if not training:
    transforms.extend([
        # (eval only) Tokenize raw solution
        gm.data.Tokenize(tokenizer=tokenizer, key="solution_raw"),
    ])

  transforms.extend([
      # 7. Pad prompt to P (raised to fit instruction + image span).
      gm.data.Pad(key="prompt", max_length=prompt_len, truncate=False),
  ])

  if not training:
    # (eval only) Pad raw response tokens for accuracy metric.
    # Also preserve the raw puzzle tokens so we can identify masked cells.
    transforms.extend([
        gm.data.Pad(key="solution_raw", max_length=256, truncate=True),
        kd.data.Elements(rename={"solution_raw": "solution_tokens"}),
        gm.data.Tokenize(tokenizer=tokenizer, key="puzzle_raw"),
        gm.data.Pad(key="puzzle_raw", max_length=256, truncate=True),
        kd.data.Elements(rename={"puzzle_raw": "puzzle_tokens"}),
    ])

  transforms.extend([
      # 8. Canvas Chunker (unchanged — canvases never contain image tokens)
      adapter_data.CanvasChunker(
          in_response="response",
          out_canvas="canvas",
          out_canvas_id="canvas_id",
          out_canvas_mask="canvas_mask",
          num_canvases=num_canvases,
          canvas_size=canvas_size,
          eos_token=eos_token,
          pad_token=pad_token,
      ),
      # 9. Sequence Target Shift + zero encoder_target_mask on the whole
      #    image span (\n\n <soi> slots <eoi> \n\n).
      vision_data.VisionSequenceTargetShift(
          pad_token=pad_token,
      ),
      # 10. Add trailing dimension to canvas (standard Kauldron Rearrange)
      kd.data.Rearrange(key="canvas", pattern="c -> c 1"),
  ])

  keep_fields = [
      "prompt",
      "patches",
      "positions_xy",
      "canvas",
      "canvas_id",
      "canvas_mask",
      "encoder_target",
      "encoder_target_mask",
  ]
  if not training:
    keep_fields.extend(["solution_tokens", "puzzle_tokens"])

  transforms.append(kd.data.Elements(keep=keep_fields))

  return sudoku_data.Bagz(
      bagz_path=bagz_path,
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      num_workers=num_workers,
      read_options=grain.ReadOptions(
          num_threads=16,
          prefetch_buffer_size=500,
      ),
      transforms=transforms,
      slice_start=slice_start,
      slice_stop=slice_stop,
  )
