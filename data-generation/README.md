# Distillation data generation

These scripts regenerate the reward-verified train and held-out splits bundled under each example. They import the checked-in environment modules so prompt construction, multi-turn transitions, terminal parsing, and reward verification stay aligned with training and evaluation.

## Credentials

Set the teacher API key only in the process environment:

```bash
export OPENROUTER_API_KEY="..."
```

You may instead pass `--api-env-file /path/to/private.env`. Keep that file outside this repository. The scripts read `OPENROUTER_API_KEY` at runtime and never write it to data, manifests, or logs.

Default teachers:

- Kimi K3 (`moonshotai/kimi-k3`): running total, structured number guess, math boxed, math Python, thinking math, and thinking science
- GLM-5.2 (`z-ai/glm-5.2`): Sudoku and logic boolean

The bundled corpora were generated with Kimi K2.6, which these defaults have since replaced. Regenerating produces a new K3 corpus rather than reproducing the checked-in one byte for byte; pass `--teacher-model moonshotai/kimi-k2.6` to stay on the original teacher.

The shipped logic-boolean recipe consumes its strict-normalized generated corpus for the SFT warm start before GRPO.

Use `--teacher-model` only when intentionally running a new experiment.

## Single-turn tasks

`distill.py` supports the four single-turn data sources. It downloads the frozen public source split where needed, sends only training prompts to the teacher, verifies each completion with the imported environment reward, and writes `train.jsonl`, `heldout.json`, `attempts.jsonl`, and `manifest.json`.

```bash
uv run python data-generation/distill.py \
  --task math-boxed-sft \
  --output-dir generated/math-boxed-sft \
  --train-size 150 --heldout-size 50

uv run python data-generation/distill.py \
  --task thinking-math-sft-opd \
  --output-dir generated/thinking-math-sft-opd \
  --train-size 150 --heldout-size 50

uv run python data-generation/distill.py \
  --task thinking-science-opd \
  --output-dir generated/thinking-science-opd \
  --train-size 150 --heldout-size 50

uv run python data-generation/distill.py \
  --task logic-boolean-sft-grpo \
  --output-dir generated/logic-boolean-sft-grpo \
  --train-size 150 --heldout-size 50
```

Math boxed and thinking math use deterministic samples from the official GSM8K train and test splits. Thinking science uses deterministic OpenBookQA train and test samples. Logic boolean uses disjoint seeded pools from the checked-in recursive generator.

## Multi-turn tasks

`distill_multiturn.py` drives complete native episodes and retains the ordered assistant and environment-feedback turns only when the imported environment reports success.

```bash
uv run python data-generation/distill_multiturn.py \
  --task running-total-sft \
  --output-dir generated/running-total-sft \
  --train-size 100 --heldout-size 50 \
  --generation-attempts 1

uv run python data-generation/distill_multiturn.py \
  --task structured-number-guess-sft-grpo \
  --output-dir generated/structured-number-guess-sft-grpo \
  --train-size 100 --heldout-size 50 \
  --generation-attempts 1

uv run python data-generation/distill_multiturn.py \
  --task math-python-sft \
  --output-dir generated/math-python-sft \
  --train-size 100 --heldout-size 50 \
  --generation-attempts 1 \
  --allow-unsafe-local-code-execution

uv run python data-generation/distill_multiturn.py \
  --task sudoku-sft-grpo \
  --output-dir generated/sudoku-sft-grpo \
  --train-size 100 --heldout-size 50 \
  --generation-attempts 1
```

Math Python executes teacher-generated Python directly on the generation or replay host with no sandbox. Both commands refuse to run that task without `--allow-unsafe-local-code-execution`; use the opt-in only on a disposable machine or container.

The task defaults select 64 completion tokens for running total and number guess, 384 for math Python, and 1536 for Sudoku. Sudoku stops each teacher call at `</move>` so every assistant turn contains exactly one action.

Replay a generated multi-turn artifact before using it:

```bash
uv run python data-generation/validate_multiturn_outputs.py \
  --task sudoku-sft-grpo \
  --output-dir generated/sudoku-sft-grpo
```

The validator reconstructs the frozen split, compares complete held-out records, checks disjointness and row counts, replays every retained transcript through the environment, verifies that no held-out id reached the teacher-attempt log, and scans for the active key value when one is available in the environment. Math Python replay also requires `--allow-unsafe-local-code-execution`.
