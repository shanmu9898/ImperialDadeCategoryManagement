"""Command-line entry point for the Imperial Dade pipeline.

Usage:
    imperial-dade run --category cups --stage all
    imperial-dade run --category cutlery --stage matching
    imperial-dade list-categories
    imperial-dade render-prompt --category cups
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Force UTF-8 on stdout so emojis / arrows / etc. don't crash on Windows cp1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from imperial_dade import __version__
from imperial_dade.categories import load_category

logger = logging.getLogger("imperial_dade")


STAGES = ("taxonomy", "matching", "feedback", "optimization", "report", "all")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _bootstrap() -> None:
    """Load .env (if present) before any other imports touch the env."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _configure_logging(os.getenv("IMPERIAL_DADE_LOG_LEVEL", "INFO"))


@click.group()
@click.version_option(__version__, prog_name="imperial-dade")
def cli() -> None:
    """Imperial Dade category-agnostic substitution pipeline."""
    _bootstrap()


@cli.command(name="list-categories")
def list_categories() -> None:
    """List available category configs."""
    from imperial_dade.categories.base import _CATEGORY_DIR

    found = sorted(p.stem for p in _CATEGORY_DIR.glob("*.yaml"))
    if not found:
        click.echo("No category configs found.")
        return
    for name in found:
        cfg = load_category(name)
        click.echo(f"  {name:<12}  attrs={len(cfg.matching.critical_attributes):<3}  top_n={cfg.matching.top_n}")


@cli.command(name="render-prompt")
@click.option("--category", "-c", required=True, help="Category name (e.g. cups).")
@click.option(
    "--hard-rules",
    default=None,
    help="Optional feedback rules to inject into the system prompt.",
)
def render_prompt_cmd(category: str, hard_rules: str | None) -> None:
    """Render the matching system prompt for a category and write it to stdout."""
    from imperial_dade.llm.prompts import render_prompt

    cfg = load_category(category)
    click.echo(render_prompt("matching_system.j2", category=cfg, hard_rules=hard_rules))


@cli.command()
@click.option("--category", "-c", required=True, help="Category name (e.g. cups, cutlery).")
@click.option(
    "--stage",
    "-s",
    type=click.Choice(STAGES, case_sensitive=False),
    default="all",
    help="Which stage to run (default: all).",
)
@click.option(
    "--notebook",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override the notebook template path.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the executed notebook (default: notebooks/runs/).",
)
def run(category: str, stage: str, notebook: Path | None, output_dir: Path | None) -> None:
    """Run the pipeline for a category by executing the parameterized notebook via papermill."""
    import papermill as pm

    repo_root = Path(__file__).resolve().parents[2]
    template = notebook or repo_root / "notebooks" / "pipeline_template.ipynb"
    out_root = output_dir or repo_root / "notebooks" / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_category(category)  # validates the YAML up-front
    logger.info("Running pipeline for %s (top_n=%d)", cfg.name, cfg.matching.top_n)

    out_path = out_root / f"{category.lower()}_{stage}_run.ipynb"
    pm.execute_notebook(
        input_path=str(template),
        output_path=str(out_path),
        parameters={"category": category.lower(), "stage": stage.lower()},
        log_output=True,
        progress_bar=False,
    )
    logger.info("Wrote executed notebook: %s", out_path)


def main() -> None:  # entry point used by pyproject.toml
    try:
        cli(standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except Exception as exc:  # pragma: no cover - top-level safety net
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
