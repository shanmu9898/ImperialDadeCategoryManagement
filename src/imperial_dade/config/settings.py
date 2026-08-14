"""Runtime settings sourced from environment variables.

Replaces the ad-hoc constants in legacy `OpenAI Code/pyvent.py`. All secrets
come from the process environment; load a `.env` file at the repo root with
`python-dotenv` before importing if you want file-based config in dev.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_batch_endpoint: str
    azure_batch_key: str
    data_dir: Path
    log_level: str


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "See .env.example at the repo root."
        )
    return val


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    endpoint = _require("AZURE_OPENAI_ENDPOINT")
    key = _require("AZURE_OPENAI_API_KEY")

    return Settings(
        azure_openai_endpoint=endpoint,
        azure_openai_key=key,
        azure_batch_endpoint=os.getenv("AZURE_BATCH_OPENAI_ENDPOINT") or endpoint,
        azure_batch_key=os.getenv("AZURE_BATCH_OPENAI_KEY") or key,
        data_dir=Path(os.getenv("IMPERIAL_DADE_DATA_DIR", "Data")).resolve(),
        log_level=os.getenv("IMPERIAL_DADE_LOG_LEVEL", "INFO").upper(),
    )
