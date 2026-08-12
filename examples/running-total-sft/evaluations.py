"""Why this example has no `flash env eval` suite.

`flash env eval` is single-turn: it sends one prompt, takes one response, and calls
`EvalSuite.score(case, response_text)`. Running total feeds one number per turn and grades every
running total in the transcript, so a single reply cannot produce the number this example
reports. `EnvironmentEvalSuite` rejects an `EnvironmentMultiTurn` for exactly this reason.

Scoring one reply anyway would still print a percentage, and that percentage would silently
measure a different task. Failing loudly is the point.
"""

from __future__ import annotations

from freesolo.environments import EvalSuite

_MESSAGE = (
    "running-total-sft is a multi-turn environment, and `flash env eval` scores a single "
    "response per case. Run the local multi-turn evaluator instead:\n"
    "  uv run python eval/evaluate_suite.py --example running-total-sft "
    "--output eval-results/running-total-model.json"
)


def load_evaluations(environment=None, **kwargs: object) -> list[EvalSuite]:
    raise NotImplementedError(_MESSAGE)
