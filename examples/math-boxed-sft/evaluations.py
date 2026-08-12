"""Held-out evaluation suite for `flash env eval`.

Published beside environment.py, so `flash env eval <run-id>` scores a deployed adapter against
the same 50 frozen cases eval/evaluate_suite.py uses, graded by the environment's own reward.
"""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEvalSuite, EvalSuite

_HELDOUT_PATH = Path(__file__).parent / "data" / "heldout.json"


def heldout_cases() -> list[TaskExample]:
    """The 50 frozen cases, never sent to the teacher."""
    rows = json.loads(_HELDOUT_PATH.read_text())
    cases = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        answer = metadata.get("answer", row.get("answer"))
        cases.append(
            TaskExample(
                record=row,
                id=row["id"],
                input=row["input"],
                output=f"\\boxed{{{answer}}}",
                metadata=metadata,
            )
        )
    return cases


def load_evaluations(environment=None, **kwargs: object) -> list[EvalSuite]:
    """Build the held-out suite graded by this environment's own scorer.

    `flash env eval` passes its own `FreesoloEnvironment` wrapper rather than the SDK
    environment, and `EnvironmentEvalSuite` requires the latter. Unwrap it when present so the
    suite grades with the same reward the run trains against; fall back to constructing the
    environment directly when called without one.
    """
    scorer = getattr(environment, "_env", environment) if environment is not None else None
    if scorer is None:
        from environment import load_environment

        scorer = load_environment(**kwargs)
    return [
        EnvironmentEvalSuite(scorer, heldout_cases(), name="heldout"),
    ]
