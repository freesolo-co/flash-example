# Running total with multi-turn SFT

## What it teaches

This multi-turn, non-thinking SFT example teaches a 0.8B model to maintain a running sum across three or four dialogue turns. The environment and SFT completion use the same deterministic turn builder.

## Files

- `environment.py`: dialogue lifecycle, deterministic data, gold trajectories, and exact scoring
- `train.toml`: bounded multi-turn SFT configuration
- `call.py`: deployed multi-turn evaluation
- `smoke_test.py`: network-free arithmetic checks
- `PROVENANCE.md`: source record

## Prerequisites

Install Python 3.11 or 3.12 and uv. First follow the repository root README Quickstart to install the pinned Flash dev client at commit `c669f0b4`, then authenticate the Flash CLI. The released `freesolo-flash==0.2.57` wheel currently fails production server validation, so do not install it yet. Set `FLASH_OPENAI_BASE_URL`, `FREESOLO_API_KEY`, and later `FLASH_RUN_ID` only for deployed calls.

## Create or scaffold equivalent

```bash
mkdir -p examples/running-total-sft
uv sync
```

A multi-turn example needs deterministic episode start, step, turn limit, scoring, and SFT completion methods.

## Local smoke

```bash
uv run python examples/running-total-sft/smoke_test.py
```

## Push environment

```bash
flash env push --name running-total-sft examples/running-total-sft
```

## Replace environment id

Replace `clay/running-total-sft` in `train.toml` with the returned id.

## Dry-run

```bash
flash train examples/running-total-sft/train.toml --dry-run
```

## Cost

```bash
flash train examples/running-total-sft/train.toml --cost
```

Review the estimate before starting paid work.

## Paid train

```bash
flash train examples/running-total-sft/train.toml --background
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

SFT final adapters can be deployed even when no per-step list appears. Deploy `RUN_ID` for the final adapter. If a checkpoint is listed, its exact selector is `RUN_ID/step-N`.

## Deploy with verification

```bash
flash deploy RUN_ID
```

For a listed step, use `flash deploy RUN_ID/step-N`. Exact checkpoint selection uses `RUN_ID/step-N`, while serving calls use base `RUN_ID`. Verify the deployment before evaluation.

## CLI call

```bash
flash chat RUN_ID -m "Number: 3"
```

`flash chat` sends a single request only. The complete multi-turn interaction is driven by `uv run python examples/running-total-sft/call.py`.

## Python call and evaluation

```bash
export FLASH_RUN_ID=RUN_ID
uv run python examples/running-total-sft/call.py
```

The script preserves assistant history, sends three user turns, and checks every total.

## Undeploy

```bash
flash undeploy RUN_ID
```

## Tested results

See the repository root [VALIDATION.md](../../VALIDATION.md) for the authoritative record of live training validation and the current deployment blocker for all examples.

## Provenance and limitations

See `PROVENANCE.md`. Addends and episode lengths are intentionally small. This example tests dialogue state and format adherence, not general arithmetic ability.
