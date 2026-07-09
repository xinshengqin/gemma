"""Environment compatibility shims for the ft_gemma smoke/test entry points.

The installed etils (etree/enp) still probes ``jax._src.prng.KeyTy`` to
detect PRNG-key dtypes, but jax 0.10 removed that module — any
``element_spec`` computation over a data pipeline raises AttributeError.
``patch_etils_jax_prng`` installs a stub module whose ``KeyTy`` matches
nothing, which is semantically correct here: data batches never contain
PRNG keys.

This is tooling glue for the local environment, not part of the
add-visual-inputs design.
"""

import sys
import types


def patch_etils_jax_prng() -> None:
  """Installs a ``jax._src.prng`` stub if the running jax lacks it."""
  import jax  # pylint: disable=g-import-not-at-top

  if hasattr(jax._src, "prng"):  # pylint: disable=protected-access
    return

  class _NeverKeyTy:
    """Placeholder for the removed KeyTy; no dtype is an instance of it."""

  module = types.ModuleType("jax._src.prng")
  module.KeyTy = _NeverKeyTy
  sys.modules["jax._src.prng"] = module
  jax._src.prng = module  # pylint: disable=protected-access
