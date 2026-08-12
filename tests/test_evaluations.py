"""Check every evaluations.py through the loader `flash env eval` actually uses.

Loading the sidecar is not enough: a suite that returns cases but grades everything the same
would still load, still print a percentage, and still be wrong. Each single-turn suite is
scored against its own gold output and against junk, and the two must separate.

Flash ships as a `uv tool` install rather than a dependency of this repo, so it is usually
absent from the test venv. Skip when it is missing instead of pinning it.
"""

from __future__ import annotations

import pytest

flash_evaluations = pytest.importorskip(
    "flash.envs.evaluations", reason="flash CLI is not installed in this venv"
)
flash_adapter = pytest.importorskip(
    "flash.envs.adapter", reason="flash CLI is not installed in this venv"
)

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# `flash env eval` sends one prompt and grades one response, so only single-turn tasks can use it.
SINGLE_TURN = (
    "logic-boolean-sft-grpo",
    "thinking-science-opd",
    "math-boxed-sft",
    "thinking-math-sft-opd",
)
# these grade a transcript, so their sidecar must refuse rather than score one reply.
MULTI_TURN = (
    "running-total-sft",
    "structured-number-guess-sft-grpo",
    "math-python-sft",
    "sudoku-sft-grpo",
)
EXPECTED_CASES = 50


def entrypoint(example: str) -> str:
    return str(ROOT / "examples" / example / "environment.py")


def load_suites(example: str):
    path = entrypoint(example)
    environment = flash_adapter.load_freesolo_environment(path)
    return flash_evaluations.load_evaluation_suites(path, environment=environment)


@pytest.mark.parametrize("example", SINGLE_TURN + MULTI_TURN)
def test_every_example_ships_a_sidecar(example: str) -> None:
    assert (ROOT / "examples" / example / "evaluations.py").is_file()


@pytest.mark.parametrize("example", SINGLE_TURN)
def test_single_turn_suite_exposes_the_frozen_cases(example: str) -> None:
    suites = load_suites(example)
    assert len(suites) == 1
    cases = suites[0].cases()
    assert len(cases) == EXPECTED_CASES
    assert all(case.id for case in cases), "every case needs an id to report against"


@pytest.mark.parametrize("example", SINGLE_TURN)
def test_single_turn_suite_separates_gold_from_junk(example: str) -> None:
    """The suite must reward the gold answer and reject junk.

    A scorer that returns a constant passes a 'suite loads' check but measures nothing.
    """
    suite = load_suites(example)[0]
    cases = suite.cases()
    gold = [suite.score(case, str(case.output)).score for case in cases]
    junk = [suite.score(case, "nonsense").score for case in cases]
    assert min(gold) == 1.0, f"{example}: a gold answer scored below 1.0"
    assert max(junk) == 0.0, f"{example}: junk scored above 0.0"


@pytest.mark.parametrize("example", MULTI_TURN)
def test_multi_turn_sidecar_refuses_and_names_the_local_runner(example: str) -> None:
    """Refusing is the correct behavior, but only if the message says what to run instead."""
    with pytest.raises(Exception) as excinfo:
        load_suites(example)
    message = str(excinfo.value)
    assert "eval/evaluate_suite.py" in message
    assert example in message
