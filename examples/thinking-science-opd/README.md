# Science distillation with OPD

This example uses single-stage OPD with Kimi K2.6 to train `Qwen/Qwen3.5-9B` on exact multiple-choice science answers. The directory name is retained from the original example, but the shipped algorithm is OPD, not GRPO.

The bundled offline Kimi K2.6 corpus and OpenBookQA held-out split are included for audit and reuse. The shipped `train.toml` performs online OPD over the published environment and does not consume `data/train.jsonl`.

## Files

- `environment.py`: prompt formatting and exact final-letter reward
- `data/train.jsonl`: 135 verified Kimi K2.6 completions
- `data/heldout.json`: 50 OpenBookQA test questions never sent to the teacher
- `train.toml`: shipped 50-step OPD recipe
- `call.py`: deployed science check
- `smoke_test.py`: network-free parser checks

## Train

```bash
flash env push --name thinking-science-opd examples/thinking-science-opd
flash train examples/thinking-science-opd/train.toml --dry-run
flash train examples/thinking-science-opd/train.toml --cost
flash train examples/thinking-science-opd/train.toml --background
```

The shipped run is `flash-1784321193-dab97aef`. It answered 50/50 held-out questions correctly, matching GPT-5.5 at 50/50. A tested 4B student reached only 90%, so the 9B student is the smallest shipped model from this search.

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
