"""Detokenization summary for vision-input prompts.

The expanded prompt carries the ``-2`` soft-token placeholders
(``SOFT_TOKEN_PLACEHOLDER``), which are not real vocabulary ids — the
baseline ``DetokenizePromptAndResponse`` would crash SentencePiece with
"piece id is out of range" when decoding them. This variant strips the
negative sentinel ids from the prompt before detokenization; everything else
is unchanged.
"""

from __future__ import annotations

import dataclasses

import flax.struct
from gemma.diffusion.hackable_diffusion_adapter.eval import text_metric
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True)
class VisionDetokenizePromptAndResponse(
    text_metric.DetokenizePromptAndResponse
):
  """Detokenize prompt and response, skipping the soft-token placeholders."""

  @flax.struct.dataclass
  class State(text_metric.DetokenizePromptAndResponse.State):
    """Collects the first num_texts prompt+response pairs."""

    def compute(self) -> list[str]:
      """Detokenizes collected prompts and responses, pairing them together."""
      results = []
      for p, r in zip(self.prompt, self.response):
        p = np.asarray(p)
        # Drop the -2 soft-token placeholders (and any other sentinel ids):
        # they mark where the image embeddings were merged and have no text.
        p = p[p >= 0]
        prompt_text = self.parent.tokenizer.decode(p.tolist())
        response_text = self.parent.tokenizer.decode(np.asarray(r).tolist())
        results.append(prompt_text + self.parent.separator + response_text)
      return results
