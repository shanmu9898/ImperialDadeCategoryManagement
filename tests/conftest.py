"""Shared pytest fixtures.

Sets dummy env vars so imports that touch `get_settings()` don't fail when
the developer runs tests without a `.env` file.
"""
from __future__ import annotations

import os

# Inject before any imperial_dade.* import happens.
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://placeholder.openai.azure.com")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "placeholder")


import pandas as pd
import pytest

from imperial_dade.categories.base import _CATEGORY_DIR


@pytest.fixture(scope="session")
def available_categories() -> list[str]:
    """Every category YAML present in the package."""
    return sorted(p.stem for p in _CATEGORY_DIR.glob("*.yaml"))


@pytest.fixture
def taxonomy_fixture() -> pd.DataFrame:
    """A 100-row taxonomy-shaped DataFrame.

    Mirrors the columns Taxonomy stage consumes: an Entity--Item id,
    a description, vendor/customer attributes, and the cost columns
    referenced in PipelineConfig.TRANSACTION_COLUMNS.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "Entity--Item": [f"E{i:04d}--I{i:04d}" for i in range(n)],
        "Item Desc 1": [f"Test cup description {i}" for i in range(n)],
        "Item Desc 2": [f"detail {i}" for i in range(n)],
        "VGN": [f"VENDOR-{i % 7}" for i in range(n)],
        "VPN": [f"VPN-{i:05d}" for i in range(n)],
        "VB Flag": rng.choice(["Y - VB", "N"], size=n).tolist(),
        "Qty": rng.integers(1, 1000, size=n).tolist(),
        "Gross Cost": rng.uniform(10, 500, size=n).round(2).tolist(),
        "Net Cost": rng.uniform(5, 450, size=n).round(2).tolist(),
        "Case Pack": rng.choice([100, 250, 500, 1000, 2500], size=n).tolist(),
        "POD": rng.choice(["Y", "N"], size=n).tolist(),
        "Whs Code": [f"WH{i % 5}" for i in range(n)],
        "Customer Class": rng.choice(["Redistributor", "Direct", "National"], size=n).tolist(),
    })
