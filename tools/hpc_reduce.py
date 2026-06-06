"""Compatibility shim that aliases ``tools.hpc_reduce`` to ``rfm_pipeline.hpc_reduce``.

The reducer implementation lives in the installable package so it can be
exposed as a console entry point (``rfm-hpc-reduce``) for PyPI users. This
shim preserves the historical ``python tools/hpc_reduce.py`` invocation used
by the pixi ``rfm-hpc-reduce`` task and the ``from tools import hpc_reduce``
import pattern used by the test suite, by aliasing this module's
``sys.modules`` entry to the package implementation so that any attribute
mutation (including ``pytest.MonkeyPatch.setattr``) operates on the real
implementation.
"""

from __future__ import annotations

import sys

import rfm_pipeline.hpc_reduce as _impl

sys.modules[__name__] = _impl

if __name__ == "__main__":
    _impl.main()
