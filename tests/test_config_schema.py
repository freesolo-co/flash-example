"""Parse every shipped train config through Flash's own schema.

The configs in this repo are only useful if `flash train` accepts them. Flash's config surface
moves (keys get renamed, scoped per algorithm, or handed over to the platform), and a config that
drifted stayed silent here until a real submit rejected it. Parsing with the installed Flash turns
that into a local failure.

Flash ships as a `uv tool` install rather than a dependency of this repo, so it is usually absent
from the test venv. Skip when it is missing instead of pinning it -- the examples must keep working
against the released CLI, not a vendored copy of it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

flash_schema = pytest.importorskip("flash.schema", reason="flash CLI is not installed in this venv")

ROOT = Path(__file__).resolve().parents[1]

# every config a user is told to run, in the order the recipes run them.
CONFIG_PATHS = (
    "running-total-sft/train.toml",
    "logic-boolean-sft-grpo/train_sft.toml",
    "logic-boolean-sft-grpo/train_grpo.toml",
    "structured-number-guess-sft-grpo/train_sft.toml",
    "structured-number-guess-sft-grpo/train_grpo.toml",
    "thinking-science-opd/train.toml",
    "math-boxed-sft/train.toml",
    "math-python-sft/train.toml",
    "thinking-math-sft-opd/train_sft.toml",
    "thinking-math-sft-opd/train_opd.toml",
    "sudoku-sft-grpo/train_sft.toml",
    "sudoku-sft-grpo/train_grpo.toml",
)

# the placeholder every checked-in config carries in place of a real project uuid.
PLACEHOLDER_PROJECT = "00000000-0000-0000-0000-000000000000"


def load_raw(rel_path: str) -> dict:
    with (ROOT / "examples" / rel_path).open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.parametrize("rel_path", CONFIG_PATHS)
def test_config_parses_under_flash_schema(rel_path: str) -> None:
    """`flash train` parses the config, project included."""
    # project_required=True is what the real train and --cost paths use.
    flash_schema.spec_from_dict(load_raw(rel_path), project_required=True)


@pytest.mark.parametrize("rel_path", CONFIG_PATHS)
def test_config_carries_replaceable_project_placeholder(rel_path: str) -> None:
    """The project id is the placeholder, so nobody inherits another org's project."""
    assert load_raw(rel_path)["project"] == PLACEHOLDER_PROJECT


@pytest.mark.parametrize("rel_path", CONFIG_PATHS)
def test_optimizer_batch_uses_the_key_for_its_algorithm(rel_path: str) -> None:
    """sft sizes on batch_size; grpo/opd size on prompts_per_step. They are different quantities.

    Flash rejects the wrong name outright, so this only guards against a config that names neither
    and silently falls back to the recipe default.
    """
    raw = load_raw(rel_path)
    train = raw.get("train", {})
    expected = "batch_size" if raw["algorithm"] == "sft" else "prompts_per_step"
    forbidden = "prompts_per_step" if expected == "batch_size" else "batch_size"
    assert expected in train, f"{rel_path} does not set {expected}"
    assert forbidden not in train, f"{rel_path} sets {forbidden}, which {raw['algorithm']} rejects"
