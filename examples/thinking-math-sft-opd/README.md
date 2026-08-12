# Thinking math with SFT and OPD

This example distills Kimi K2.6 reasoning into `Qwen/Qwen3.5-9B` with SFT, then warm-starts OPD from the SFT adapter. Responses end with exactly `Answer: <integer>`.

## Files

- `environment.py`: deterministic smoke generator, bundled-data loader, exact parser, and reward
- `data/train.jsonl`: 143 verified GSM8K teacher completions used by SFT
- `data/heldout.json`: 50 held-out GSM8K test questions never sent to the teacher
- `train_sft.toml`: shipped SFT stage
- `train_opd.toml`: shipped OPD stage with Kimi K2.6 and `init_from_adapter`
- `call.py`: deployed exact-answer check
- `smoke_test.py`: network-free parser checks

## Train

Run the stages in order. Replace `init_from_adapter` in the OPD config with your new SFT run id when reproducing.

```bash
flash env push --name thinking-math-sft-opd --project <your-uuid> examples/thinking-math-sft-opd
flash train examples/thinking-math-sft-opd/train_sft.toml --dry-run
flash train examples/thinking-math-sft-opd/train_sft.toml --cost
flash train examples/thinking-math-sft-opd/train_sft.toml --background

flash train examples/thinking-math-sft-opd/train_opd.toml --dry-run
flash train examples/thinking-math-sft-opd/train_opd.toml --cost
flash train examples/thinking-math-sft-opd/train_opd.toml --background
```

Shipped runs: SFT `flash-1784325488-7a1d31b9`, then OPD `flash-1784326094-ab33c65b`. The final OPD adapter scored 0.92, matching GPT-5.5 at 0.92. The SFT-only adapter scored 0.94; OPD is retained here to show the complete warm-start workflow, not because it improved quality.

See [RESULTS.md](../../RESULTS.md). Grade a deployed adapter with `flash env eval <run-id>`.
