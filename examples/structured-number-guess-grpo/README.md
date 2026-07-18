# Structured number guess with SFT and GRPO

This multi-turn example distills Kimi K2.6 binary-search trajectories into `Qwen/Qwen3.5-2B`, then warm-starts GRPO from the SFT adapter. Every assistant turn must be a strict JSON object such as `{"guess":42}`.

## Files

- `environment.py`: bundled-data loader, strict parser, dialogue lifecycle, SFT completion, and reward
- `data/train.jsonl`: 100 verified seven-turn teacher trajectories
- `data/heldout.json`: 50 disjoint number ranges
- `train_sft.toml`: shipped SFT stage
- `train_grpo.toml`: shipped GRPO stage with `init_from_adapter`
- `call.py`: deployed seven-turn evaluator
- `smoke_test.py`: network-free parser and generator checks

## Train

Run the stages in order. The checked-in GRPO config names the shipped SFT run; replace `init_from_adapter` with your new SFT run id when reproducing.

```bash
flash env push --name structured-number-guess-grpo examples/structured-number-guess-grpo
flash train examples/structured-number-guess-grpo/train_sft.toml --dry-run
flash train examples/structured-number-guess-grpo/train_sft.toml --cost
flash train examples/structured-number-guess-grpo/train_sft.toml --background

flash train examples/structured-number-guess-grpo/train_grpo.toml --dry-run
flash train examples/structured-number-guess-grpo/train_grpo.toml --cost
flash train examples/structured-number-guess-grpo/train_grpo.toml --background
```

Shipped runs: SFT `flash-1784319868-8f8ee7be`, then GRPO `flash-1784321381-dbe68392`. Their prior held-out and ablation numbers are superseded after correcting the per-case JSON schema and regenerated secret distribution. Results are pending Phase 2 re-evaluation.

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
