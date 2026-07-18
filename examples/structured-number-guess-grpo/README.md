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

Run the stages in order. Replace `init_from_adapter` in the GRPO config with your new SFT run id when reproducing. The config retains the original campaign parent id; the corrected Phase 2 run used the new SFT parent listed below.

```bash
flash env push --name structured-number-guess-grpo examples/structured-number-guess-grpo
flash train examples/structured-number-guess-grpo/train_sft.toml --dry-run
flash train examples/structured-number-guess-grpo/train_sft.toml --cost
flash train examples/structured-number-guess-grpo/train_sft.toml --background

flash train examples/structured-number-guess-grpo/train_grpo.toml --dry-run
flash train examples/structured-number-guess-grpo/train_grpo.toml --cost
flash train examples/structured-number-guess-grpo/train_grpo.toml --background
```

Corrected Phase 2 runs: SFT `flash-1784349216-ba05be67`, then GRPO `flash-1784349640-ceb35b18`. On 50 genuine varied-secret binary-search cases, the adapter solved 50/50 with mean reward 1.0, matching GPT-5.5 at 50/50 and mean reward 1.0. The older claim that this example exceeded GPT-5.5 came from trivial constant-offset data and is retracted.

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
