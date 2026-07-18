# Boxed-answer math with SFT

This example distills Kimi K2.6 reasoning into `Qwen/Qwen3.5-9B` with pure SFT. Responses may explain their work, but the final answer must be inside the last balanced `\boxed{...}` expression. The directory name is retained from the original example, but the shipped algorithm is SFT.

## Files

- `environment.py`: deterministic smoke problems, bundled-data loader, boxed parser, and normalized reward
- `data/train.jsonl`: 143 reward-verified GSM8K teacher completions
- `data/heldout.json`: 50 held-out GSM8K test questions never sent to the teacher
- `train.toml`: shipped 75-step SFT recipe
- `call.py`: deployed boxed-answer check
- `smoke_test.py`: network-free parser and scoring checks
- `PROVENANCE.md`: source adaptation record

## Train

```bash
flash env push --name math-boxed-sft examples/math-boxed-sft
flash train examples/math-boxed-sft/train.toml --dry-run
flash train examples/math-boxed-sft/train.toml --cost
flash train examples/math-boxed-sft/train.toml --background
```

The shipped run is `flash-1784263689-f98515ce`. It reached 0.90 held-out accuracy versus GPT-5.5 at 0.92, one of 50 held-out cases behind and therefore near parity.

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
