"""Held-out evaluation suite for `flash env eval`.

Published beside environment.py, so `flash env eval <run-id>` plays each held-out case out as a
full episode and grades it with the environment's own `score_episode`, the same reward the run
trains against.

This example is multi-turn: one number arrives per turn and the reward reads every running total
in the transcript. `flash env eval` drives those turns for a suite that sets
`grades_episodes = True`, and hands the finished rollout state to `score`.
"""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets import TaskExample
from freesolo.environments import EvalSuite

_HELDOUT_PATH = Path(__file__).parent / "data" / "heldout.json"


def heldout_rows() -> list[dict]:
    """The frozen held-out rows, never used for training."""
    return json.loads(_HELDOUT_PATH.read_text())


def heldout_cases() -> list:
    """The held-out cases in the shape `flash env eval` validates.

    `flash.envs.evaluations.validate_evaluation_cases` requires real `EvalCase` instances via
    `isinstance`, so a structurally identical local class does not pass. Flash is always importable
    where this matters, because flash is the process running `env eval`. When it is absent -- the
    SDK alone, or any consumer without the CLI installed -- fall back to `TaskExample`, which
    carries the same four fields this suite needs.

    The metadata must survive the trip: the driver rebuilds the example from `case.metadata`
    (`flash/cli/commands/env/test.py:_evaluation_example`), and these environments read `numbers`,
    `secret`, `answer`, and `puzzle` from it. Dropping it fails every case with a KeyError.
    """
    rows = heldout_rows()
    try:
        from flash.envs.evaluations import EvalCase
    except ImportError:
        return [
            TaskExample(
                record=row,
                id=row["id"],
                input=row["input"],
                output=row.get("output"),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in rows
        ]
    return [
        EvalCase(
            input=row["input"],
            expected=row.get("output"),
            id=row["id"],
            # `metadata` is carried BOTH flat and nested under a "metadata" key. The driver
            # flattens `case.metadata` into the example dict, and the SDK then rebuilds
            # `TaskExample.metadata` from a NESTED `record["metadata"]` (`normalize_record`), so a
            # flat-only copy arrives as `{}` and the environment raises KeyError on its own task
            # fields. The flat copy is kept because the driver's example dict is also read
            # directly.
            metadata={**metadata, "metadata": dict(metadata)},
        )
        for row, metadata in ((row, dict(row.get("metadata") or {})) for row in rows)
    ]


def _example(case) -> dict:
    """The case as the environment's scorer expects it.

    Mirrors `_evaluation_example` in flash's own driver: metadata first, then input/output/id on
    top. The environments read their task fields (`numbers`, `secret`, `answer`, `puzzle`) off
    this dict, so a case whose metadata was dropped fails scoring with a KeyError.
    """
    metadata = getattr(case, "metadata", None) or {}
    example = dict(metadata)
    example["input"] = case.input
    example["output"] = getattr(case, "expected", None) or getattr(case, "output", None)
    if getattr(case, "id", None) is not None:
        example["id"] = case.id
    return example


class EpisodeEvalSuite:
    """Grade a played-out episode with the environment's own `score_episode`.

    `EnvironmentEvalSuite` cannot serve here: it requires an `EnvironmentSingleTurn` and its
    `score(case, response_text)` sees only the last reply, which for this task is one running
    total out of several. Scoring that would still print a percentage, and the percentage would
    measure a different task than the one being trained.

    `grades_episodes = True` is what tells `flash env eval` to play the turns instead of sending
    one prompt. It is opt-in per suite rather than implied by the environment, because a
    multi-turn environment can legitimately carry a suite that grades only the opening action.
    """

    grades_episodes = True

    def __init__(self, environment, cases, *, name: str = "heldout") -> None:
        self.environment = environment
        self._cases = list(cases)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def cases(self) -> list[TaskExample]:
        return list(self._cases)

    def score(self, case, response_text: str, state=None) -> float:
        """Score the finished episode.

        `state` is the rollout the driver just played out. Grading goes through the environment's
        `reward(completion, example, state)`, which routes multi-turn state to `score_episodes` --
        the same call `flash env test` makes, so eval and test agree by construction rather than
        by a second transcript-assembly implementation kept in sync by hand.

        Returns a float rather than a `RewardResult`: the eval contract accepts `EvalResult`,
        `float`, or `bool`, and `EvalResult` lives in flash, which an environment package does not
        depend on. A `RewardResult` here fails every case with a scoring TypeError.
        """
        if state is None:
            # No episode to grade. Raising beats returning 0.0, which would be indistinguishable
            # from a model that answered every turn wrong.
            raise RuntimeError(
                "no episode state: flash env eval did not drive this case as an episode. "
                "This suite sets grades_episodes = True and requires a flash that honours it."
            )
        score, error, _ = self.environment.reward_with_error(response_text, _example(case), state)
        if error:
            # The environment's scorer itself failed. Surfacing it keeps a broken grader from
            # being reported as a confident zero.
            raise RuntimeError(f"environment scorer failed: {error}")
        return float(score)


def load_evaluations(environment=None, **kwargs: object) -> list[EvalSuite]:
    """Build the held-out episode suite graded by this environment's own scorer.

    Unlike the single-turn examples, the Flash wrapper is kept rather than unwrapped: it owns the
    rollout-state-to-episode conversion (`_score_episode` -> `score_episodes`) that turns the
    driver's state dict into the `EnvironmentEpisode` the SDK scorer expects.
    """
    scorer = environment
    if scorer is None:
        from environment import load_environment

        scorer = load_environment(**kwargs)
    if not hasattr(scorer, "reward_with_error"):
        # the raw SDK environment exposes score_episode, not the wrapper's rollout-state api, so a
        # suite built on it would raise AttributeError on the first case instead of here. flash
        # always passes its wrapper; this only fires for a direct call that omits it.
        raise TypeError(
            f"{type(scorer).__name__} is the raw SDK environment, which cannot score a rollout "
            "state. Pass the Flash-wrapped environment: "
            "load_evaluations(environment=flash.envs.loader.load_freesolo_environment(path))"
        )
    return [EpisodeEvalSuite(scorer, heldout_cases())]
