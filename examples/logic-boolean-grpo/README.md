# Logic boolean expressions with GRPO

This example trains `Qwen/Qwen3.5-4B` with single-stage GRPO to evaluate nested boolean expressions and end with exactly `<answer>True</answer>` or `<answer>False</answer>`.

GLM-5.2 generated the bundled offline distillation corpus because Kimi K2.6 had low verified yield on the strict terminal format. The shipped recipe is pure GRPO over the deterministic environment, so `data/train.jsonl` is included for audit and reuse rather than consumed by `train.toml`.

## Files

- `environment.py`: seeded expression generator, exact parser, and reward
- `data/train.jsonl`: 150 reward-verified GLM-5.2 completions
- `data/heldout.json`: 50 disjoint generated expressions
- `train.toml`: shipped 50-step GRPO recipe
- `call.py`: deployed exact-answer check
- `smoke_test.py`: network-free generation and parser checks
- `PROVENANCE.md`: source adaptation record

## Train

```bash
flash env push --name logic-boolean-grpo examples/logic-boolean-grpo
flash train examples/logic-boolean-grpo/train.toml --dry-run
flash train examples/logic-boolean-grpo/train.toml --cost
flash train examples/logic-boolean-grpo/train.toml --background
```

The shipped run is `flash-1784316952-a904a84d`. It scored 50/50 on held-out expressions, matching GPT-5.5 at 50/50.

## Evaluate and deploy

```bash
flash deploy flash-1784316952-a904a84d
export FLASH_RUN_ID=flash-1784316952-a904a84d
uv run python examples/logic-boolean-grpo/call.py
flash undeploy flash-1784316952-a904a84d
```

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
