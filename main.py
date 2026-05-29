"""Convenience entry point for the DARFMMKL demo.

The main implementation is kept in ``DASMKL.py`` for consistency with the
original experiment scripts.
"""

import runpy


if __name__ == "__main__":
    runpy.run_module("DASMKL", run_name="__main__")
