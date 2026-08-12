# Held-out evaluation

Every example is scored on the 50 frozen rows in its `data/heldout.json`, graded by the environment's own reward. There are two ways to run that, and which one applies depends on whether the task takes one turn or several.

## Which evaluator

**`flash env eval`** is the integrated path. It scores a deployed adapter against the environment's published `evaluations.py` suite and records the result in the dashboard, so the number lands next to the run that produced it:

```bash
flash env eval <run-id>
```

It sends one prompt and grades one response, so it fits the four single-turn examples: `logic-boolean-sft-grpo`, `thinking-science-opd`, `math-boxed-sft`, and `thinking-math-sft-opd`.

**`evaluate_suite.py`** is the local path and drives the full episode loop. The other four examples are multi-turn — running total feeds one number per turn, number guess needs higher/lower feedback, math Python needs an executed tool turn, and Sudoku plays up to 30 moves — so a single response cannot produce the number they report. Their `evaluations.py` raises with a pointer here rather than scoring one reply and quietly reporting a different measurement.

Useful options that only exist locally: `--dry-run`, `--model` / `--base-url`, and the GPT-5.5 comparison below.

## Shipped adapters

> **Security warning:** `math-python-sft` executes model-generated Python directly on the evaluation host with no sandbox. The harness refuses to run it by default. Use only a disposable machine or container and pass `--allow-unsafe-local-code-execution` to acknowledge the risk. The timeout and output cap limit accidental resource use but do not make execution safe.

`evaluate_suite.py` defaults to the shipped run id for the selected example. It reads `FREESOLO_API_KEY`, or the authenticated Flash CLI config, and calls the configured OpenAI-compatible serving endpoint.

```bash
uv run python eval/evaluate_suite.py \
  --example running-total-sft \
  --output eval-results/running-total-model.json
```

Use `--model` to evaluate another base model or adapter, and `--base-url` to target a different serving endpoint. Inspect the frozen cases without making network requests:

```bash
uv run python eval/evaluate_suite.py \
  --example sudoku-sft-grpo \
  --output eval-results/sudoku-plan.json \
  --dry-run
```

For Sudoku, deploy `flash-1784327754-e1d3a602/step-8` before evaluation. Serving calls still use the base run id `flash-1784327754-e1d3a602`.

Math Python requires the explicit unsafe opt-in:

```bash
uv run python eval/evaluate_suite.py \
  --example math-python-sft \
  --allow-unsafe-local-code-execution \
  --output eval-results/math-python-model.json
```

## GPT-5.5 comparison

`evaluate_gpt55.py` sends the same held-out rows through a local OpenAI-compatible gateway with model id `gpt-5.5`. The default gateway is `http://127.0.0.1:8317/v1`; override it with `GPT55_BASE_URL` or `--base-url`.

```bash
uv run python eval/evaluate_gpt55.py \
  --example running-total-sft \
  --output eval-results/running-total-gpt55.json
```

The GPT-5.5 math Python path has the same unsafe-execution gate as the shipped-adapter evaluator and requires `--allow-unsafe-local-code-execution` on a disposable machine or container.

The harness stores per-case prompts, responses, finish reasons, native environment rewards, and aggregate success rates. Do not commit gateway credentials or raw private request logs.

## Full matrix

Run both commands once for each of these names:

- `running-total-sft`
- `logic-boolean-sft-grpo`
- `structured-number-guess-sft-grpo`
- `thinking-science-opd`
- `math-boxed-sft`
- `math-python-sft`
- `thinking-math-sft-opd`
- `sudoku-sft-grpo`

The completed campaign results are summarized in [RESULTS.md](../RESULTS.md). Token counts come from provider-reported usage and latency was measured on different serving stacks, so neither should be treated as a controlled hardware benchmark.
