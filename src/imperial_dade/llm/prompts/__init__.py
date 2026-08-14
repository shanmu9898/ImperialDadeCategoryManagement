"""Jinja2-rendered prompt registry.

Templates live as `.j2` files in this directory. `render_prompt(name, **vars)`
loads + renders them with autoescape off (we're producing plain text, not HTML).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

_PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPT_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_prompt(template_name: str, **variables) -> str:
    """Render a prompt template.

    Args:
        template_name: Filename inside `llm/prompts/`, e.g. "matching_system.j2".
        **variables:  Bindings the template references.

    Raises:
        jinja2.UndefinedError if the template references a variable not supplied.
    """
    template = _env().get_template(template_name)
    return template.render(**variables)
