"""Unit tests for the vision data transforms.

Intended behavior (design §4 / §10 steps 1-4):

  * ``ExpandImagePlaceholders`` replaces the single ``<|image|>`` id (258880)
    with ``[\n\n(108), <soi>(255999), -2 x S_v, <eoi>(258882), \n\n(108)]``.
  * ``image_span_mask`` marks each contiguous ``-2`` run PLUS the two marker
    tokens on each side; multiple images produce multiple isolated spans.
  * ``VisionSequenceTargetShift`` produces the baseline targets unchanged and
    zeroes ``encoder_target_mask`` wherever the current OR next position lies
    inside the image span — soft slots AND markers excluded from the AR loss,
    text before/after the span still supervised.
  * ``PreprocessAndPatchifyImage`` resizes to the patch budget, patchifies,
    pads with ``positions_xy = -1``, and reports ``S_v = real_patches / 9``.
"""

from absl.testing import absltest
from ft_gemma.diffusion.hackable_diffusion_adapter.data import vision_data
from gemma.diffusion.hackable_diffusion_adapter.data import data as adapter_data
from gemma.gm.nn.gemma4.vision import _preprocessing
import numpy as np

NN = 108  # \n\n
SOI = 255999  # <start_of_image>
EOI = 258882  # <end_of_image>
IMG = 258880  # <|image|> placeholder
SOFT = -2


class ExpandImagePlaceholdersTest(absltest.TestCase):

  def test_canonical_usage(self):
    """[2, 11, <|image|>, 12] with S_v=3 -> the placeholder becomes a span."""
    transform = vision_data.ExpandImagePlaceholders()
    features = transform.map({
        'prompt': np.array([2, 11, IMG, 12], dtype=np.int32),
        'soft_token_count': 3,
    })
    np.testing.assert_array_equal(
        features['prompt'],
        #  2   11  \n\n  <soi>  -2    -2    -2   <eoi>  \n\n  12
        [2, 11, NN, SOI, SOFT, SOFT, SOFT, EOI, NN, 12],
    )

  def test_expansion_length_is_s_v_plus_3(self):
    """One placeholder becomes an S_v+4 span: net +S_v+3 tokens."""
    transform = vision_data.ExpandImagePlaceholders()
    for s_v in (1, 5, 280):
      features = transform.map({
          'prompt': np.array([2, IMG], dtype=np.int32),
          'soft_token_count': s_v,
      })
      self.assertLen(features['prompt'], 2 + s_v + 3)


class ImageSpanMaskTest(absltest.TestCase):

  def test_canonical_usage(self):
    """The -2 run plus two marker tokens on each side is the span."""
    #          0  1  2   3    4     5     6    7   8  9
    prompt = [2, 5, NN, SOI, SOFT, SOFT, EOI, NN, 6, 0]
    span = vision_data.image_span_mask(np.array(prompt, dtype=np.int32))
    np.testing.assert_array_equal(
        span, [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    )

  def test_two_images_two_isolated_spans(self):
    prompt = [2, NN, SOI, SOFT, EOI, NN, 5, NN, SOI, SOFT, EOI, NN, 6]
    span = vision_data.image_span_mask(np.array(prompt, dtype=np.int32))
    np.testing.assert_array_equal(
        span, [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    )

  def test_no_image_no_span(self):
    span = vision_data.image_span_mask(np.array([2, 5, 6], dtype=np.int32))
    self.assertFalse(span.any())


class VisionSequenceTargetShiftTest(absltest.TestCase):

  def test_canonical_usage(self):
    """One toy example with the full inputs and outputs written out."""
    features = vision_data.VisionSequenceTargetShift().map({
        # position:  0  1  2   3    4     5     6    7   8  9(pad)
        'prompt': np.array(
            [2, 5, NN, SOI, SOFT, SOFT, EOI, NN, 6, 0], dtype=np.int32
        ),
        # canvas positions 10..13; last slot is an invalid (PAD) canvas slot.
        'canvas': np.array([7, 8, 1, 0], dtype=np.int32),
        'canvas_mask': np.array([True, True, True, False]),
    })

    # Targets: the full sequence (prompt ++ canvas) shifted left by one.
    np.testing.assert_array_equal(
        features['encoder_target'],
        # pos:  0  1   2    3     4     5    6   7  8  9  10 11 12 13
        [5, NN, SOI, SOFT, SOFT, EOI, NN, 6, 0, 7, 8, 1, 0, 0],
    )

    # Loss mask: position i supervised iff token i AND token i+1 are valid
    # (baseline rule) AND neither position i nor i+1 lies in the image span
    # [2..7] (vision rule).
    np.testing.assert_array_equal(
        features['encoder_target_mask'],
        [
            1,  # 0: bos(2) -> 5            text before the span
            0,  # 1: 5 -> \n\n              next position enters the span
            0,  # 2: \n\n                   in span
            0,  # 3: <soi>                  in span
            0,  # 4: -2 soft slot           in span
            0,  # 5: -2 soft slot           in span
            0,  # 6: <eoi>                  in span
            0,  # 7: \n\n                   in span
            0,  # 8: 6 -> PAD               next token invalid (baseline)
            0,  # 9: PAD position           invalid (baseline)
            1,  # 10: canvas 7 -> 8         supervised
            1,  # 11: canvas 8 -> 1 (eos)   supervised
            0,  # 12: canvas 1 -> PAD slot  next canvas slot invalid
            0,  # 13: last position         always masked
        ],
    )

  def setUp(self):
    super().setUp()
    #             0  1  2   3    4     5     6    7   8  9  10(pad)
    self.prompt = [2, 5, NN, SOI, SOFT, SOFT, EOI, NN, 6, 7, 0]
    self.canvas = [7, 8, 1, 0]
    self.canvas_mask = [True, True, True, False]
    self.features = {
        'prompt': np.array(self.prompt, dtype=np.int32),
        'canvas': np.array(self.canvas, dtype=np.int32),
        'canvas_mask': np.array(self.canvas_mask),
    }

  def test_targets_match_baseline_and_only_span_mask_zeroed(self):
    baseline = adapter_data.SequenceTargetShift().map(dict(self.features))
    vision = vision_data.VisionSequenceTargetShift().map(dict(self.features))

    # The shifted targets themselves are the baseline's, unchanged.
    np.testing.assert_array_equal(
        vision['encoder_target'], baseline['encoder_target']
    )
    # The mask differs from the baseline exactly where the current OR next
    # position lies in the image span [2..7]: positions 1..7.
    span_or_next = np.zeros(len(self.prompt) + len(self.canvas), bool)
    span_or_next[1:8] = True
    np.testing.assert_array_equal(
        vision['encoder_target_mask'],
        baseline['encoder_target_mask'] & ~span_or_next,
    )

  def test_supervision_around_the_span(self):
    mask = vision_data.VisionSequenceTargetShift().map(dict(self.features))[
        'encoder_target_mask'
    ]
    self.assertTrue(mask[0])  # bos -> text before the span: supervised
    self.assertFalse(mask[1])  # predicts \n\n (span start): masked
    self.assertFalse(mask[4])  # soft slot: masked
    self.assertFalse(mask[7])  # trailing \n\n position: masked
    self.assertTrue(mask[8])  # text after the span -> text: supervised
    self.assertFalse(mask[9])  # next is prompt PAD: masked (baseline rule)
    self.assertFalse(mask[10])  # PAD position: masked (baseline rule)
    self.assertTrue(mask[11])  # canvas tokens: supervised
    self.assertTrue(mask[12])
    self.assertFalse(mask[13])  # next canvas token invalid (baseline rule)


class PreprocessAndPatchifyImageTest(absltest.TestCase):

  MAX_SOFT_TOKENS = 4  # P_p = 36

  def test_canonical_usage(self):
    """A solid-white 20x30 image, all outputs written out.

    With a budget of 4 soft tokens (36 patches), the aspect-preserving
    resize maps 20x30 px -> 48x96 px = a 3-row x 6-column grid of 16x16
    patches (18 real patches), i.e. S_v = 18 / 9 = 2 soft tokens. A white
    image stays exactly 1.0 after the [0,1] rescale, so every real patch is
    all-ones and every padding patch all-zeros.
    """
    transform = vision_data.PreprocessAndPatchifyImage(
        max_soft_tokens=self.MAX_SOFT_TOKENS
    )
    white = np.full((20, 30, 3), 255, dtype=np.uint8)
    features = transform.map({'image': white})

    self.assertEqual(features['soft_token_count'], 2)

    # Patch grid positions, raster order (x fastest), then -1 padding.
    np.testing.assert_array_equal(
        features['positions_xy'],
        [
            # row 0 of the patch grid
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
            # row 1
            [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
            # row 2
            [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2],
            # 18 padding slots (P_p = 36)
            [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
            [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
            [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
        ],
    )

    # 18 real patches of exactly 1.0 (white), 18 padding patches of 0.0;
    # each patch is 16*16*3 = 768 values.
    self.assertEqual(features['patches'].shape, (36, 768))
    np.testing.assert_array_equal(features['patches'][:18], 1.0)
    np.testing.assert_array_equal(features['patches'][18:], 0.0)

  def test_shapes_padding_and_soft_token_count(self):
    transform = vision_data.PreprocessAndPatchifyImage(
        max_soft_tokens=self.MAX_SOFT_TOKENS
    )
    image = np.random.RandomState(0).randint(
        0, 255, (20, 30, 3), dtype=np.uint8
    )
    features = transform.map({'image': image, 'other': 1})

    patches = features['patches']
    positions = features['positions_xy']
    s_v = features['soft_token_count']

    self.assertEqual(patches.shape, (36, 768))  # [P_p, 16*16*3]
    self.assertEqual(positions.shape, (36, 2))
    self.assertEqual(patches.dtype, np.float32)
    self.assertEqual(positions.dtype, np.int32)

    # S_v matches the shared predictor used by the design.
    expected_s_v = _preprocessing.predict_soft_token_count(
        20, 30, max_soft_tokens=self.MAX_SOFT_TOKENS
    )
    self.assertEqual(s_v, expected_s_v)

    # Real patches first (positions >= 0), then -1 padding; count = 9 * S_v.
    is_real = (positions >= 0).all(axis=-1)
    self.assertEqual(int(is_real.sum()), 9 * s_v)
    self.assertTrue(is_real[: 9 * s_v].all())
    np.testing.assert_array_equal(positions[9 * s_v :], -1)
    # Padding patches are zero-valued; real ones are in [0, 1].
    np.testing.assert_array_equal(patches[9 * s_v :], 0.0)
    self.assertTrue((patches[: 9 * s_v] >= 0).all())
    self.assertTrue((patches[: 9 * s_v] <= 1).all())

    # The image key is consumed; other keys pass through.
    self.assertNotIn('image', features)
    self.assertEqual(features['other'], 1)


if __name__ == '__main__':
  absltest.main()
