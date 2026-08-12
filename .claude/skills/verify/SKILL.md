# Verify Flash example runtime paths

Use the published evaluation sidecars as the runtime surface. Do not use paid models, GPUs, or
deployed adapters for local verification.

1. Load each example through the loader `flash env eval` itself uses:
   `flash.envs.evaluations.load_evaluation_suites(<example>/environment.py, environment=...)`, then
   validate its cases with `validate_evaluation_cases`. Both must succeed for all eight examples.
2. Confirm every case is a flash `EvalCase` carrying non-empty metadata. The environments score
   from their task fields (`numbers`, `secret`, `answer`, `puzzle`), so a case that loads with
   empty metadata fails every episode instead of grading it.
3. For the four multi-turn examples, confirm the suite sets `grades_episodes = True` AND that
   `flash.cli.commands.env.episode._state_argument(suite.score)` is not None. The loader wraps
   every suite, so the opt-in and the state parameter must both survive that wrapper or the
   episode is silently graded as a single reply.
4. Drive episodes with a local stub for `_generate_case` -- never a real endpoint -- and use a
   task-appropriate oracle per example, not one generic stub: binary search for number guess, a
   fenced python block followed by a boxed answer for math python, moves replayed from the known
   solution for sudoku. A generic stub can score identically for a correct and a broken model,
   which proves nothing.
5. Require the oracle and a junk model to separate on every example. Equal scores mean the harness
   is not measuring the task, regardless of whether the numbers look plausible.
6. Probe strict reward paths with malformed logic answers and malformed sudoku moves. Note that
   sudoku moves need `<move>A1=5</move>` tags with nothing after them; a bare `A1=5` parses as no
   move at all and scores the untouched starting board.
7. Keep all outputs under `/tmp`; do not train, deploy, or call external model endpoints.
