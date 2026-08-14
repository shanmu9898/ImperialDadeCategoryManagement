"""Azure OpenAI credential surface.

Replaces legacy `OpenAI Code/pyvent.py`. No hardcoded values — everything
flows from environment via `imperial_dade.config.settings.get_settings()`.
"""
from __future__ import annotations

import os
from pathlib import Path

from imperial_dade.config.settings import get_settings

_settings = get_settings()

AZURE_OPENAI_ENDPOINT: str = _settings.azure_openai_endpoint
AZURE_OPENAI_KEY: str = _settings.azure_openai_key
AZURE_BATCH_OPENAI_ENDPOINT: str = _settings.azure_batch_endpoint
AZURE_BATCH_OPENAI_KEY: str = _settings.azure_batch_key
ROOT_DIR: str = str(Path(__file__).resolve().parents[2])


class SaveTool:
    """Lightweight stand-in for the legacy pyvent.SaveTool.

    Creates input/output/cache subdirectories under `data_path` on construction
    so callers can rely on the paths existing.
    """

    def __init__(
        self,
        data_path: str = "data",
        input_path: str = "input",
        output_path: str = "output",
        cache_path: str | None = None,
        cache_name: str = "cache",
        **kwargs,
    ):
        self.data_path = data_path
        self.input_path = os.path.join(data_path, input_path)
        self.output_path = os.path.join(data_path, output_path)
        self.cache_path = cache_path or os.path.join(data_path, "cache")
        self.cache_name = cache_name
        self.full_cache_path = os.path.join(self.cache_path, cache_name)

        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)

        for key, value in kwargs.items():
            setattr(self, key, value)


class Constants:
    AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_KEY = AZURE_OPENAI_KEY
    AZURE_BATCH_OPENAI_KEY = AZURE_BATCH_OPENAI_KEY
    AZURE_BATCH_OPENAI_ENDPOINT = AZURE_BATCH_OPENAI_ENDPOINT
    ROOT_DIR = ROOT_DIR
