"""Data input/output for the pipeline.

`fornax.FornaxLoader` is the entry point for everything the pipeline needs
to read from the central data platform. All disk-based reads in the legacy
notebooks have been replaced by Fornax queries.
"""

from imperial_dade.io.fornax import FornaxLoader, get_fornax_connection

__all__ = ["FornaxLoader", "get_fornax_connection"]
