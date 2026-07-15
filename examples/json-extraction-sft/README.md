# JSON extraction with SFT

## What it teaches

This single-turn, non-thinking SFT example teaches a 0.8B model to map short customer messages to strict JSON labels with `category`, `priority`, and `summary`. SFT learns from JSON gold completions. Flash 0.2.57 does not apply `train.structured_outputs` to SFT, so serving-time calls pass `response_format` explicitly.

## Files

- `environment.py`: deterministic synthetic data, prompts, gold completions, and exact JSON scoring
- `train.toml`: bounded SFT configuration
- `call.py`: deployed call with a strict JSON schema
- `smoke_test.py`: network-free helper checks
- `PROVENANCE.md`: source record

## Prerequisites

Install Python 3.11 or 3.12 and uv. First follow the repository root README Quickstart to install the pinned Flash dev client at commit `c669f0b4`, then authenticate the Flash CLI. The released `freesolo-flash==0.2.57` wheel currently fails production server validation, so do not install it yet. Set `FLASH_OPENAI_BASE_URL`, `FREESOLO_API_KEY`, and later `FLASH_RUN_ID` only when calling a deployment.

## Create or scaffold equivalent

```bash
mkdir -p examples/json-extraction-sft
uv sync
```

Keep the environment, training config, deployment call, smoke test, and provenance record together.

## Local smoke

```bash
uv run python examples/json-extraction-sft/smoke_test.py
```

## Push environment

```bash
flash env push --name json-extraction-sft examples/json-extraction-sft
```

## Replace environment id

Copy the returned environment id into `[environment].id` in `train.toml`, replacing `clay/json-extraction-sft`.

## Dry-run

```bash
flash train examples/json-extraction-sft/train.toml --dry-run
```

## Cost

```bash
flash train examples/json-extraction-sft/train.toml --cost
```

Review the estimate before starting paid work.

## Paid train

```bash
flash train examples/json-extraction-sft/train.toml --background
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

SFT final adapters can be deployed even when `flash checkpoints` has no per-step list. Use `RUN_ID` for the final adapter. If a step is listed, exact selection uses `RUN_ID/step-N`.

## Deploy with verification

```bash
flash deploy RUN_ID
```

If selecting a listed checkpoint, run `flash deploy RUN_ID/step-N`. Deployment selection may use `RUN_ID/step-N`, but serving calls and `FLASH_RUN_ID` use the base `RUN_ID`. Verify deployment health before evaluation.

## CLI call

```bash
flash chat RUN_ID -m "I was charged twice for order 18."
```

This sends one customer message for JSON extraction.

## Python call and evaluation

```bash
export FLASH_RUN_ID=RUN_ID
uv run python examples/json-extraction-sft/call.py
```

The script passes `response_format` and rejects missing or extra fields.

## Undeploy

```bash
flash undeploy RUN_ID
```

## Tested results

See the repository root [VALIDATION.md](../../VALIDATION.md) for the authoritative record of live training validation and the current deployment blocker for all examples.

## Provenance and limitations

See `PROVENANCE.md`. The dataset is small and synthetic. It demonstrates mechanics, not production coverage. Serving-time schemas improve format validity but do not guarantee semantic correctness.
