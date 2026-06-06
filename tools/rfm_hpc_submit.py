"""Thin shim — delegates to the installed rfm_pipeline.hpc_submit module.

Kept here so that `pixi run rfm-hpc-submit` (which calls this file directly)
continues to work without any changes to pixi.toml.
"""

from rfm_pipeline.hpc_submit import main

if __name__ == "__main__":
    main()
