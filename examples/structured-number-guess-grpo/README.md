# Structured number guess with multi-turn GRPO

## What it teaches

This multi-turn, non-thinking GRPO example teaches binary-search-like interaction. Every model turn must be strict JSON such as `{"guess":42}`. Both training rollouts and deployed calls use the same JSON schema.

## Files

- `environment.py`: deterministic secrets, strict parser, dialogue lifecycle, and reward
- `train.toml`: bounded GRPO config with `train.structured_outputs`
- `call.py`: deployed seven-turn evaluator with `response_format`
- `smoke_test.py`: network-free parser and data checks
- `PROVENANCE.md`: source record

## Prerequisites

Install Python 3.11 or 3.12 and uv. First follow the repository root README Quickstart to install the pinned Flash dev client at commit `c669f0b4`, then authenticate the Flash CLI. The released `freesolo-flash==0.2.57` wheel currently fails production server validation, so do not install it yet. Set `FLASH_OPENAI_BASE_URL`, `FREESOLO_API_KEY`, and later `FLASH_RUN_ID` only for deployed calls.

## Create or scaffold equivalent

```bash
mkdir -p examples/structured-number-guess-grpo
uv sync
```

Keep the schema, parser, reward, and deployed response format aligned.

## Local smoke

```bash
uv run python examples/structured-number-guess-grpo/smoke_test.py
```

## Push environment

```bash
flash env push --name structured-number-guess-grpo examples/structured-number-guess-grpo
```

## Replace environment id

Replace `clay/structured-number-guess-grpo` in `train.toml` with the returned id.

## Dry-run

```bash
flash train examples/structured-number-guess-grpo/train.toml --dry-run
```

## Cost

```bash
flash train examples/structured-number-guess-grpo/train.toml --cost
```

Review rollout count and cost before paid work.

## Paid train

```bash
flash train examples/structured-number-guess-grpo/train.toml --background
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

GRPO can expose step checkpoints. Use `RUN_ID/step-N` to select an exact listed checkpoint, or `RUN_ID` for the final adapter.

## Deploy with verification

```bash
flash deploy RUN_ID/step-N
```

You may instead deploy `RUN_ID`. Exact selection uses `RUN_ID/step-N`, but serving calls and `FLASH_RUN_ID` use base `RUN_ID`. Verify the deployment before evaluation.

## CLI call

```bash
flash chat RUN_ID -m "Find the secret number from 1 through 100."
```

`flash chat` sends a single request only. The complete multi-turn interaction is driven by `uv run python examples/structured-number-guess-grpo/call.py`.

## Python call and evaluation

```bash
export FLASH_RUN_ID=RUN_ID
uv run python examples/structured-number-guess-grpo/call.py
```

The script passes `response_format`, drives every dialogue turn, and requires the model to find 42 within seven turns.

## Undeploy

```bash
flash undeploy RUN_ID
```

## Tested results

See the repository root [VALIDATION.md](../../VALIDATION.md) for the authoritative record of live training validation and the current deployment blocker for all examples.

## Provenance and limitations

See `PROVENANCE.md`. Structured decoding guarantees the JSON shape during configured generation, not an efficient search strategy. The tiny fixed range is educational.
