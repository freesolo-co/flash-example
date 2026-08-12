"""Why this example has no `flash env eval` suite.

`flash env eval` is single-turn: it sends one prompt, takes one response, and calls
`EvalSuite.score(case, response_text)`. Math Python only earns credit after an executed Python
tool turn, so the tool call and its result must both be in the transcript -- a single reply cannot
produce the number this example reports. `EnvironmentEvalSuite` rejects an `EnvironmentMultiTurn`
for exactly this reason.

Scoring one reply anyway would still print a percentage, and that percentage would silently
measure a different task. Failing loudly is the point.
"""

from __future__ import annotations

from freesolo.environments import EvalSuite

_MESSAGE = (
    "math-python-sft is a multi-turn tool-use environment, and `flash env eval` scores a single "
    "response per case. Run the local multi-turn evaluator instead (it executes model-written "
    "Python, so use a disposable machine or container):\n"
    "  uv run python eval/evaluate_suite.py --example math-python-sft "
    "--allow-unsafe-local-code-execution --output eval-results/math-python-model.json"
)


def load_evaluations(environment=None, **kwargs: object) -> list[EvalSuite]:
    raise NotImplementedError(_MESSAGE)
