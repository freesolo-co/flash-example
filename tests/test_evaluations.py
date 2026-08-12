"""Check every evaluations.py through the loader `flash env eval` actually uses.

Loading the sidecar is not enough: a suite that returns cases but grades everything the same
would still load, still print a percentage, and still be wrong. Each single-turn suite is
scored against its own gold output and against junk, and the two must separate.

The multi-turn suites cannot be checked that way here -- their score depends on a played-out
episode, which needs a deployed model. What is checked is everything that must hold BEFORE the
model is involved: the opt-in survives the loader's wrapper, the scorer still advertises the
state parameter, and the cases carry the metadata the environment scores from. Those are the
parts that failed silently, each turning a gradeable episode into a uniform wrong number.

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

# these send one prompt and grade one response.
SINGLE_TURN = (
    "logic-boolean-sft-grpo",
    "thinking-science-opd",
    "math-boxed-sft",
    "thinking-math-sft-opd",
)
# these grade a transcript. `flash env eval` plays the episode out for a suite that sets
# `grades_episodes = True`, so they are scored rather than refused.
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
def test_multi_turn_suite_opts_into_episode_grading(example: str) -> None:
    """The suite must ask to be driven, and expose the state parameter that makes it meaningful.

    `grades_episodes` alone is a promise. The driver decides whether to pass the finished episode
    by inspecting the scorer's signature, and the loader wraps every suite before that inspection,
    so both have to survive the trip or the suite is handed one reply and grades the wrong task.
    """
    episode = pytest.importorskip("flash.cli.commands.env.episode")

    suite = load_suites(example)[0]
    assert episode._grades_episodes(suite) is True
    assert episode._state_argument(suite.score) is not None


@pytest.mark.parametrize("example", MULTI_TURN)
def test_multi_turn_suite_exposes_the_frozen_cases(example: str) -> None:
    """Cases must be the type flash validates, carrying the metadata the environment scores from.

    `validate_evaluation_cases` requires real `EvalCase` objects, and the environments read their
    task fields (`numbers`, `secret`, `answer`, `puzzle`) out of the case metadata. A case that
    loads but drops either one fails every episode instead of grading it.
    """
    flash_evals = pytest.importorskip("flash.envs.evaluations")

    suite = load_suites(example)[0]
    cases = flash_evals.validate_evaluation_cases(suite, source=entrypoint(example))
    assert len(cases) == EXPECTED_CASES
    assert all(case.id for case in cases), "every case needs an id to report against"
    assert all(case.metadata for case in cases), "the environment scores from case metadata"
