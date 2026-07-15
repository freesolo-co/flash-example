# Thinking science with GRPO

## What it teaches

This single-turn, thinking-enabled GRPO example trains exact final-letter answers on a small set of original multiple-choice science questions. It intentionally omits `structured_outputs` because thinking structured serving is not generally available in Flash 0.2.57.

## Files

- `environment.py`: original questions, prompt formatting, and exact final-letter reward
- `train.toml`: bounded thinking GRPO configuration
- `call.py`: deployed science evaluation
- `smoke_test.py`: network-free parser and data checks
- `PROVENANCE.md`: source record

## Prerequisites

Install Python 3.11 or 3.12 and uv. First follow the repository root README Quickstart to install the pinned Flash dev client at commit `c669f0b4`, then authenticate the Flash CLI. The released `freesolo-flash==0.2.57` wheel currently fails production server validation, so do not install it yet. Set `FLASH_OPENAI_BASE_URL`, `FREESOLO_API_KEY`, and later `FLASH_RUN_ID` only for deployed calls.

## Create or scaffold equivalent

```bash
mkdir -p examples/thinking-science-grpo
uv sync
```

Write questions and answer keys together, and keep the final answer parser narrow.

## Local smoke

```bash
uv run python examples/thinking-science-grpo/smoke_test.py
```

## Push environment

```bash
flash env push --name thinking-science-grpo examples/thinking-science-grpo
```

## Replace environment id

Replace `clay/thinking-science-grpo` in `train.toml` with the returned id.

## Dry-run

```bash
flash train examples/thinking-science-grpo/train.toml --dry-run
```

## Cost

```bash
flash train examples/thinking-science-grpo/train.toml --cost
```

Review rollout count and cost before paid work.

## Paid train

```bash
flash train examples/thinking-science-grpo/train.toml --background
```

## Monitor

```bash
flash status RUN_ID --follow
flash log RUN_ID --follow
```

## Checkpoint or final adapter choice

```bash
flash checkpoints RUN_ID
```

GRPO can expose step checkpoints. Use `RUN_ID/step-N` for an exact listed checkpoint or `RUN_ID` for the final adapter.

## Deploy with verification

```bash
flash deploy RUN_ID/step-N
```

You may instead deploy `RUN_ID`. Exact selection uses `RUN_ID/step-N`, while serving calls and `FLASH_RUN_ID` use base `RUN_ID`. Verify deployment before evaluation.

## CLI call

```bash
flash chat RUN_ID -m "What force pulls objects toward Earth? A) Magnetism B) Friction C) Gravity D) Buoyancy"
```

This sends one multiple-choice science question and requests a final `Answer: <letter>` line.

## Python call and evaluation

```bash
export FLASH_RUN_ID=RUN_ID
uv run python examples/thinking-science-grpo/call.py
```

The script scores one original question by exact final answer letter. It does not pass `response_format`.

## Undeploy

```bash
flash undeploy RUN_ID
```

## Tested results

See the repository root [VALIDATION.md](../../VALIDATION.md) for the authoritative record of live training validation and the current deployment blocker for all examples.

## Provenance and limitations

See `PROVENANCE.md`. The question set is small, repeated to form a bounded training pool, and not a science benchmark. Exact-letter reward does not assess explanation quality.
