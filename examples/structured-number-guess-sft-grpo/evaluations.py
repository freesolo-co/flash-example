"""Why this example has no `flash env eval` suite.

`flash env eval` is single-turn: it sends one prompt, takes one response, and calls
`EvalSuite.score(case, response_text)`. Number guess needs the feedback loop -- the model guesses,
the environment answers higher or lower -- so a single reply cannot produce the number this
example reports. `EnvironmentEvalSuite` rejects an `EnvironmentMultiTurn` for exactly this reason.

Scoring one reply anyway would still print a percentage, and that percentage would silently
measure a different task. Failing loudly is the point.
"""

from __future__ import annotations

from freesolo.environments import EvalSuite

_MESSAGE = (
    "structured-number-guess-sft-grpo is a multi-turn environment, and `flash env eval` scores a "
    "single response per case. Run the local multi-turn evaluator instead:\n"
    "  uv run python eval/evaluate_suite.py --example structured-number-guess-sft-grpo "
    "--output eval-results/number-guess-model.json"
)


def load_evaluations(environment=None, **kwargs: object) -> list[EvalSuite]:
    raise NotImplementedError(_MESSAGE)
