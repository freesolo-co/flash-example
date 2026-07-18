# Logic boolean expressions with SFT and GRPO

This example trains `Qwen/Qwen3.5-4B` to evaluate nested boolean expressions and end with exactly `<answer>True</answer>` or `<answer>False</answer>`. The shipped recipe uses GLM-5.2 trajectories for an SFT warm start, then applies GRPO with a strict all-or-nothing reward.

The strict parser accepts either a bare terminal answer or one leading `<think>...</think>` block followed by exactly one terminal `<answer>...</answer>`. The bundled 150-row SFT dataset is normalized to that contract and parser-verified.

## Files

- `environment.py`: seeded expression generator, bundled-data loader, exact parser, and reward
- `data/train.jsonl`: 150 strict-normalized, reward-verified GLM-5.2 completions
- `data/heldout.json`: 50 disjoint generated expressions
- `train_sft.toml`: shipped 100-step SFT stage
- `train_grpo.toml`: shipped 50-step GRPO stage with `init_from_adapter`
- `call.py`: deployed exact-answer check
- `smoke_test.py`: network-free generation and parser checks
- `PROVENANCE.md`: source adaptation record

## Train

Run the stages in order. Replace `init_from_adapter` in the GRPO config with your new SFT run id when reproducing.

```bash
flash env push --name logic-boolean-sft-grpo examples/logic-boolean-sft-grpo
flash train examples/logic-boolean-sft-grpo/train_sft.toml --dry-run
flash train examples/logic-boolean-sft-grpo/train_sft.toml --cost
flash train examples/logic-boolean-sft-grpo/train_sft.toml --background

flash train examples/logic-boolean-sft-grpo/train_grpo.toml --dry-run
flash train examples/logic-boolean-sft-grpo/train_grpo.toml --cost
flash train examples/logic-boolean-sft-grpo/train_grpo.toml --background
```

Shipped runs: SFT `flash-1784350210-5225a6a4`, then GRPO `flash-1784350975-6dc07970`. The SFT adapter scored 45/50, and the GRPO adapter scored 48/50 with mean reward 0.96 versus GPT-5.5 at 50/50. Both remaining misses were deeply nested expressions whose correct value was `True`. A 9B SFT-to-GRPO run missed the same two cases, so the 4B adapter is shipped.

Single-stage GRPO scored 0/50. Under this strict all-or-nothing reward, cold-start rollouts that do not satisfy the exact terminal format receive zero reward, leaving no useful gradient. The SFT stage establishes the output contract before GRPO.

## Evaluate and deploy

```bash
flash deploy flash-1784350975-6dc07970
export FLASH_RUN_ID=flash-1784350975-6dc07970
uv run python examples/logic-boolean-sft-grpo/call.py
flash undeploy flash-1784350975-6dc07970
```

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
