"""Convert Sudoku data to the *visual* Bagz format (add-visual-inputs design).

Each record holds the puzzle rendered as one image of the 9x9 grid (the
puzzle digits live only in the image) plus the solution text:

  - ``puzzle_image``: PNG bytes of the rendered grid.
  - ``solution``: space-separated solved grid string.
  - ``puzzle``: space-separated puzzle string (kept for the baseline Sudoku
    eval metrics, which need the puzzle tokens to identify masked cells).

Two input modes:
  * ``--input_csv``: the same Kaggle Sudoku CSV the text converter uses
    (columns ``quizzes,solutions`` or ``puzzle,solution``).
  * ``--fake_records=N``: generate N random fake examples (random digit
    grids; NOT valid Sudokus) — for smoke tests only.
"""

import csv
import io
import os
import random

from absl import app
from absl import flags
from absl import logging
import bagz
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf

_INPUT_CSV = flags.DEFINE_string(
    "input_csv",
    "",
    "Path to the input Sudoku CSV file.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Directory to write the output Bagz files.",
)
_TRAIN_SPLIT = flags.DEFINE_float(
    "train_split",
    0.9,
    "Fraction of data to use for training.",
)
_MAX_RECORDS = flags.DEFINE_integer(
    "max_records",
    -1,
    "Maximum number of records to process. Use -1 for all.",
)
_FAKE_RECORDS = flags.DEFINE_integer(
    "fake_records",
    0,
    "If > 0, generate this many random fake examples instead of reading"
    " --input_csv (smoke tests only).",
)
_SEED = flags.DEFINE_integer(
    "seed",
    0,
    "Random seed for --fake_records.",
)

_GRID_SIZE = 9
_CELL_PX = 28


def render_sudoku_image(puzzle: str, cell_px: int = _CELL_PX) -> np.ndarray:
  """Renders a Sudoku puzzle string as an image of the 9x9 grid.

  Empty cells (0) are left blank. Thick lines mark the 3x3 boxes.

  Args:
    puzzle: 81 digits, optionally space-separated ('0' = empty cell).
    cell_px: Pixel size of one cell.

  Returns:
    uint8 RGB array of shape [9*cell_px + 3, 9*cell_px + 3, 3].
  """
  digits = puzzle.split() if " " in puzzle else list(puzzle)
  if len(digits) != _GRID_SIZE * _GRID_SIZE:
    raise ValueError(
        f"Expected {_GRID_SIZE * _GRID_SIZE} digits, got {len(digits)}"
    )

  side = _GRID_SIZE * cell_px + 3
  image = Image.new("RGB", (side, side), "white")
  draw = ImageDraw.Draw(image)

  for i in range(_GRID_SIZE + 1):
    width = 3 if i % 3 == 0 else 1
    offset = i * cell_px
    draw.line([(offset, 0), (offset, side)], fill="black", width=width)
    draw.line([(0, offset), (side, offset)], fill="black", width=width)

  for row in range(_GRID_SIZE):
    for col in range(_GRID_SIZE):
      digit = digits[row * _GRID_SIZE + col]
      if digit == "0":
        continue
      x = col * cell_px + cell_px // 2
      y = row * cell_px + cell_px // 2
      draw.text((x, y), digit, fill="black", anchor="mm")

  return np.asarray(image, dtype=np.uint8)


def encode_png(image: np.ndarray) -> bytes:
  buf = io.BytesIO()
  Image.fromarray(image).save(buf, format="PNG")
  return buf.getvalue()


def make_tf_example(features_dict: dict[str, bytes]) -> tf.train.Example:
  """Create a tf.train.Example proto from a dictionary of byte features."""
  tf_features = {
      k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
      for k, v in features_dict.items()
  }
  return tf.train.Example(features=tf.train.Features(feature=tf_features))


def space_separate(s: str) -> str:
  """Insert space characters between every character in the string."""
  return " ".join(list(s))


def make_vision_record(puzzle: str, solution: str) -> bytes:
  """Builds one serialized visual-Sudoku record from raw digit strings."""
  puzzle_spaced = space_separate(puzzle)
  solution_spaced = space_separate(solution)
  image = render_sudoku_image(puzzle_spaced)
  features = {
      "puzzle_image": encode_png(image),
      "puzzle": puzzle_spaced.encode("utf-8"),
      "solution": solution_spaced.encode("utf-8"),
  }
  return make_tf_example(features).SerializeToString()


def make_fake_examples(
    num_records: int, seed: int = 0
) -> list[tuple[str, str]]:
  """Generates random (puzzle, solution) digit strings for smoke tests.

  The grids are random digits, NOT valid Sudokus — sufficient to exercise
  the data pipeline and training step.

  Args:
    num_records: Number of examples to generate.
    seed: Random seed.

  Returns:
    List of (puzzle, solution) 81-char digit strings.
  """
  rng = random.Random(seed)
  examples = []
  for _ in range(num_records):
    solution = "".join(str(rng.randint(1, 9)) for _ in range(81))
    puzzle = "".join(
        "0" if rng.random() < 0.5 else digit for digit in solution
    )
    examples.append((puzzle, solution))
  return examples


def write_vision_bagz(records: list[tuple[str, str]], path: str) -> None:
  """Renders and writes (puzzle, solution) records to a visual Bagz file."""
  with bagz.Writer(path) as writer:
    for puzzle, solution in records:
      writer.write(make_vision_record(puzzle, solution))
  logging.info("Wrote %d records to %s", len(records), path)


def _read_csv(path: str) -> list[tuple[str, str]]:
  records = []
  with open(path, newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    puzzle_key = "quizzes" if "quizzes" in fields else "puzzle"
    solution_key = "solutions" if "solutions" in fields else "solution"
    for row in reader:
      records.append((row[puzzle_key], row[solution_key]))
  return records


def main(argv):
  del argv
  out_dir = _OUTPUT_DIR.value
  os.makedirs(out_dir, exist_ok=True)

  if _FAKE_RECORDS.value > 0:
    records = make_fake_examples(_FAKE_RECORDS.value, seed=_SEED.value)
  else:
    records = _read_csv(_INPUT_CSV.value)

  if _MAX_RECORDS.value > 0:
    records = records[: _MAX_RECORDS.value]

  split = int(len(records) * _TRAIN_SPLIT.value)
  train_records, eval_records = records[:split], records[split:]

  write_vision_bagz(
      train_records, os.path.join(out_dir, "sudoku_vision_train.bagz")
  )
  write_vision_bagz(
      eval_records, os.path.join(out_dir, "sudoku_vision_eval.bagz")
  )


if __name__ == "__main__":
  app.run(main)
