# Tool-using Python math with SFT

This multi-turn example distills Kimi K2.6 tool-use trajectories into `Qwen/Qwen3.5-4B` with pure SFT. The model writes Python in fenced blocks, receives captured output, and finishes with a `\boxed{...}` answer. The directory name is retained from the original example, but the shipped algorithm is SFT.

## Files

- `environment.py`: bundled-data loader, Python executor, dialogue lifecycle, SFT completion, and exact reward
- `data/train.jsonl`: 97 verified teacher tool-use trajectories
- `data/heldout.json`: 50 disjoint generated arithmetic problems
- `train.toml`: shipped 75-step SFT recipe
- `call.py`: deployed multi-turn evaluator
- `smoke_test.py`: network-free tool and reward checks
- `PROVENANCE.md`: source adaptation record

## Train

```bash
flash env push --name math-python-sft --project <your-uuid> examples/math-python-sft
flash train examples/math-python-sft/train.toml --dry-run
flash train examples/math-python-sft/train.toml --cost
flash train examples/math-python-sft/train.toml --background
```

The shipped run is `flash-1784322317-e152ffdd`. It scored 92% reward accuracy versus GPT-5.5 at 82% under the strict tool-use and boxed-answer contract.

## Execution safety

The local executor uses a temporary directory, a five-second wall-clock timeout, bounded captured output, and a minimal child environment. It still executes model-generated Python directly on the host and is not a security sandbox. Run it only on a disposable machine or inside a container, microVM, or remote code sandbox. The evaluation harness requires `--allow-unsafe-local-code-execution` before enabling this path.

See [RESULTS.md](../../RESULTS.md) and [eval](../../eval).
