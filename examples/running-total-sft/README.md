# Running total with multi-turn SFT

This example distills Kimi K2.6 trajectories into `Qwen/Qwen3.5-2B` with pure SFT. The model must return the exact running total after each turn as a bare integer.

## Files

- `environment.py`: deterministic generator, bundled-data loader, multi-turn lifecycle, SFT completion, and exact reward
- `data/train.jsonl`: 100 verified teacher trajectories used by the shipped SFT run
- `data/heldout.json`: 50 disjoint evaluation episodes never sent to the teacher
- `train.toml`: shipped 100-step SFT recipe
- `call.py`: deployed multi-turn check
- `smoke_test.py`: network-free arithmetic checks

## Train

The checked-in environment id is the immutable campaign environment containing the same bundled training data. Replace it if you publish your own copy.

```bash
flash env push --name running-total-sft examples/running-total-sft
flash train examples/running-total-sft/train.toml --dry-run
flash train examples/running-total-sft/train.toml --cost
flash train examples/running-total-sft/train.toml --background
```

The shipped run is `flash-1784320041-9c4a32b8`. It scored 100% on 50 held-out episodes, matching GPT-5.5 at 100%.

## Evaluate and deploy

```bash
flash deploy flash-1784320041-9c4a32b8
export FLASH_RUN_ID=flash-1784320041-9c4a32b8
uv run python examples/running-total-sft/call.py
flash undeploy flash-1784320041-9c4a32b8
```

See [RESULTS.md](../../RESULTS.md) for the campaign comparison and [eval](../../eval) for the held-out harness.
