# pyvent.py
# Minimal standalone replacement for pyvent library dependencies

import os


# =============================================
# pyvent.tools.save_tools
# =============================================

class SaveTool:
    """Enhanced replacement for pyvent's SaveTool with additional parameters"""

    def __init__(self, data_path="data", input_path="input", output_path="output",
                 cache_path=None, cache_name="cache", **kwargs):
        self.data_path = data_path
        self.input_path = os.path.join(data_path, input_path)
        self.output_path = os.path.join(data_path, output_path)

        # Handle cache parameters
        self.cache_path = cache_path or os.path.join(data_path, "cache")
        self.cache_name = cache_name
        self.full_cache_path = os.path.join(self.cache_path, cache_name)

        # Create directories if they don't exist
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)

        # Store any additional kwargs for compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================
# pyvent.tools (module structure)
# =============================================

class Tools:
    class SaveTools:
        SaveTool = SaveTool


tools = Tools()

# =============================================
# pyvent.constants
# =============================================

# Azure OpenAI configuration — read from environment only.
# See .env.example at the repo root for required variables.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_BATCH_OPENAI_KEY = os.getenv("AZURE_BATCH_OPENAI_KEY", AZURE_OPENAI_KEY)
AZURE_BATCH_OPENAI_ENDPOINT = os.getenv("AZURE_BATCH_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT)

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    raise RuntimeError(
        "Azure OpenAI credentials missing. Set AZURE_OPENAI_API_KEY and "
        "AZURE_OPENAI_ENDPOINT in your environment (see .env.example)."
    )

# Add ROOT_DIR for compatibility if needed
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================
# Module structure to match pyvent imports
# =============================================

class Constants:
    AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_KEY = AZURE_OPENAI_KEY
    AZURE_BATCH_OPENAI_KEY = AZURE_BATCH_OPENAI_KEY
    AZURE_BATCH_OPENAI_ENDPOINT = AZURE_BATCH_OPENAI_ENDPOINT
    ROOT_DIR = ROOT_DIR


constants = Constants()