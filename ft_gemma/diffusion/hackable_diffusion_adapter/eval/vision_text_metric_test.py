"""Unit tests for ``VisionDetokenizePromptAndResponse``.

Intended behavior: the expanded vision prompt carries ``-2`` soft-token
placeholders, which are not vocabulary ids — the baseline summary crashes
SentencePiece on them ("piece id is out of range"). The vision variant
strips the sentinels before detokenization and is otherwise identical.

Needs network access on first use (the Gemma4 tokenizer model is fetched
from GCS), like the rest of the tokenizer-dependent tests.
"""

from absl.testing import absltest
from ft_gemma.diffusion.hackable_diffusion_adapter.eval import vision_text_metric
from gemma.diffusion.hackable_diffusion_adapter.eval import text_metric
import numpy as np

# "hello world"-ish plain ids plus BOS; exact ids don't matter, only that
# they are valid vocabulary entries.
_TEXT_IDS = [2, 76857, 506, 29344]
_RESPONSE = np.array([[506, 29344, 1]], dtype=np.int32)


def _prompt_with_placeholders():
  return np.array(
      [[2, 76857, 108, 255999, -2, -2, -2, 258882, 108, 506]], dtype=np.int32
  )


class VisionDetokenizeTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.metric = vision_text_metric.VisionDetokenizePromptAndResponse(
        prompt='batch.prompt', response='samples', num_texts=2
    )
    cls.baseline = text_metric.DetokenizePromptAndResponse(
        prompt='batch.prompt', response='samples', num_texts=2
    )

  def test_strips_soft_token_placeholders(self):
    """Decodes an expanded prompt; result == decoding without sentinels."""
    prompt = _prompt_with_placeholders()
    texts = self.metric.get_state(
        prompt=prompt, response=_RESPONSE
    ).compute()

    prompt_no_sentinels = prompt[prompt >= 0][None, :]
    expected = self.metric.get_state(
        prompt=prompt_no_sentinels, response=_RESPONSE
    ).compute()

    self.assertEqual(texts, expected)
    self.assertLen(texts, 1)
    self.assertIsInstance(texts[0], str)

  def test_plain_text_prompt_matches_baseline(self):
    """Without placeholders the vision variant is the baseline."""
    prompt = np.array([_TEXT_IDS], dtype=np.int32)
    vision_texts = self.metric.get_state(
        prompt=prompt, response=_RESPONSE
    ).compute()
    baseline_texts = self.baseline.get_state(
        prompt=prompt, response=_RESPONSE
    ).compute()
    self.assertEqual(vision_texts, baseline_texts)

  def test_baseline_crashes_on_placeholders(self):
    """Documents why the override exists: -2 is not a decodable piece id."""
    state = self.baseline.get_state(
        prompt=_prompt_with_placeholders(), response=_RESPONSE
    )
    with self.assertRaises(Exception):
      state.compute()


if __name__ == '__main__':
  absltest.main()
