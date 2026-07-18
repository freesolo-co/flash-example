# Sudoku with SFT and GRPO

This multi-turn example uses GLM-5.2 trajectories to warm-start `Qwen/Qwen3.5-4B` with SFT, then applies GRPO over the exact Sudoku environment. GLM-5.2 replaced Kimi K2.6 because it had higher verified yield under the one-move terminal contract.

## Files

- `environment.py`: deterministic unique-puzzle generator, bundled-data loader, board lifecycle, SFT completion, and shaped reward
- `data/train.jsonl`: 99 verified GLM-5.2 solve trajectories
- `data/heldout.json`: 50 disjoint puzzles never sent to the teacher
- `train_sft.toml`: shipped SFT stage
- `train_grpo.toml`: shipped GRPO stage with `init_from_adapter`
- `call.py`: complete deployed episode evaluator
- `smoke_test.py`: network-free full-solve checks
- `PROVENANCE.md`: source adaptation record

## Train

Run the stages in order. Replace `init_from_adapter` in the GRPO config with your new SFT run id when reproducing.

```bash
flash env push --name sudoku-grpo examples/sudoku-grpo
flash train examples/sudoku-grpo/train_sft.toml --dry-run
flash train examples/sudoku-grpo/train_sft.toml --cost
flash train examples/sudoku-grpo/train_sft.toml --background

flash train examples/sudoku-grpo/train_grpo.toml --dry-run
flash train examples/sudoku-grpo/train_grpo.toml --cost
flash train examples/sudoku-grpo/train_grpo.toml --background
```

Shipped runs: SFT `flash-1784324917-1d8f0ce8`, then GRPO `flash-1784327754-e1d3a602`, with step 8 selected. Under the strict corrected reward, the adapter solved 49/50 with mean shaped reward 5.5714 versus GPT-5.5 at 50/50 and 5.6700. This is near parity, one case behind. The earlier held-out and ablation numbers remain superseded because they predated clue immutability, strict one-move parsing, and unique-solution scoring.

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
