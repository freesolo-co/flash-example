# Thinking math with OPD

## What it teaches

This single-turn, thinking-enabled OPD example distills synthetic grade-school word-problem behavior into a 0.8B model. The reward checks only the exact final numeric line, while `teacher_model = "glm-5.2"` supplies dense teacher signal during training.

## Files

- `environment.py`: deterministic synthetic problems and exact final-number reward
- `train.toml`: bounded thinking OPD configuration
- `call.py`: deployed arithmetic evaluation
- `smoke_test.py`: network-free parser and data checks
- `PROVENANCE.md`: source record

## Prerequisites

Install Python 3.11 or 3.12 and uv. First follow the repository root README Quickstart to install the pinned Flash dev client at commit `c669f0b4`, then authenticate the Flash CLI. The released `freesolo-flash==0.2.57` wheel currently fails production server validation, so do not install it yet. The OPD teacher is managed by Flash. Set `FLASH_OPENAI_BASE_URL`, `FREESOLO_API_KEY`, and later `FLASH_RUN_ID` only for deployed calls.

## Create or scaffold equivalent

```bash
mkdir -p examples/thinking-math-opd
uv sync
```

Keep the problem generator and exact answer contract deterministic so evaluation remains reproducible.

## Local smoke

```bash
uv run python examples/thinking-math-opd/smoke_test.py
```

## Push environment

```bash
flash env push --name thinking-math-opd examples/thinking-math-opd
```

## Replace environment id

Replace `clay/thinking-math-opd` in `train.toml` with the returned id.

## Dry-run

```bash
flash train examples/thinking-math-opd/train.toml --dry-run
```

## Cost

```bash
flash train examples/thinking-math-opd/train.toml --cost
```

OPD includes teacher scoring cost. Review the estimate before paid work.

## Paid train

```bash
flash train examples/thinking-math-opd/train.toml --background
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

OPD can expose step checkpoints. Use `RUN_ID/step-N` for an exact listed checkpoint or `RUN_ID` for the final adapter.

## Deploy with verification

```bash
flash deploy RUN_ID/step-N
```

You may instead deploy `RUN_ID`. Exact selection uses `RUN_ID/step-N`, while serving calls and `FLASH_RUN_ID` use base `RUN_ID`. Verify deployment before evaluation.

## CLI call

```bash
flash chat RUN_ID -m "Mina has 4 marbles and receives 3 more. How many marbles does she have now?"
```

This sends one short word problem and requests a final `Answer: <integer>` line.

## Python call and evaluation

```bash
export FLASH_RUN_ID=RUN_ID
uv run python examples/thinking-math-opd/call.py
```

The script evaluates one original multiplication problem by exact final number.

## Undeploy

```bash
flash undeploy RUN_ID
```

## Tested results

See the repository root [VALIDATION.md](../../VALIDATION.md) for the authoritative record of live training validation and the current deployment blocker for all examples.

## Provenance and limitations

See `PROVENANCE.md`. The data is newly synthetic and narrow. Exact-answer scoring ignores reasoning quality. The bounded two-step config is a smoke, not a quality recipe.
