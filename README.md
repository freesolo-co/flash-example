# Flash training examples

Eight end-to-end examples for training small Qwen3.5 task adapters with [Flash](https://github.com/freesolo-co/flash). All eight use teacher supervision through bundled SFT trajectories or OPD, and all eight now have frozen 50-case evaluations under their active contracts.

This is not a general claim that the students beat GPT-5.5. Some students score higher on frozen tasks because GPT-5.5 occasionally violates strict boxed-answer, tool-use, or multi-turn contracts. See [RESULTS.md](RESULTS.md) for the full framing, run ids, footprint notes, corrected-result history, and SFT-versus-RL ablations.

## Examples

| Example                                                          | Base       | Shipped recipe   | Teacher   | Held-out result              |
| ---------------------------------------------------------------- | ---------- | ---------------- | --------- | ---------------------------- |
| [Running total](examples/running-total-sft)                      | Qwen3.5-2B | pure SFT         | Kimi K2.6 | 100%, GPT-5.5 100%           |
| [Logic boolean](examples/logic-boolean-sft-grpo)                     | Qwen3.5-4B | SFT to GRPO      | GLM-5.2   | 48/50, GPT-5.5 50/50         |
| [Structured number guess](examples/structured-number-guess-sft-grpo) | Qwen3.5-2B | SFT to GRPO      | Kimi K2.6 | 50/50, GPT-5.5 50/50         |
| [Thinking science](examples/thinking-science-opd)               | Qwen3.5-9B | single-stage OPD | Kimi K2.6 | 50/50, GPT-5.5 50/50         |
| [Math boxed](examples/math-boxed-sft)                           | Qwen3.5-9B | pure SFT         | Kimi K2.6 | 0.90, GPT-5.5 0.92           |
| [Math Python](examples/math-python-sft)                         | Qwen3.5-4B | pure SFT         | Kimi K2.6 | 92%, GPT-5.5 82%             |
| [Thinking math](examples/thinking-math-sft-opd)                      | Qwen3.5-9B | SFT to OPD       | Kimi K2.6 | 0.92, GPT-5.5 0.92           |
| [Sudoku](examples/sudoku-sft-grpo)                                   | Qwen3.5-4B | SFT to GRPO      | GLM-5.2   | 49/50, GPT-5.5 50/50, strict |

Kimi K2.6 is `moonshotai/kimi-k2.6`. Six recipes use Kimi K2.6 supervision, and logic boolean plus Sudoku use `z-ai/glm-5.2` trajectories. Logic boolean now consumes its strict-normalized GLM-5.2 corpus for SFT before GRPO.

Recipe coverage is three pure-SFT examples (running total, math Python, and math boxed), one single-stage OPD example (thinking science), three SFT-to-GRPO examples (structured number guess, Sudoku, and logic boolean), and one SFT-to-OPD example (thinking math). There is no pure single-stage-GRPO example. Logic's strict all-or-nothing reward cold-started single-stage GRPO at 0/50, so the shipped recipe uses an SFT warm start.

The four community-inspired environments retain their provenance files: math boxed and math Python are adapted from Prime Intellect math environments, logic boolean from `primeintellect/logic-env`, and Sudoku from `m8ngotree/sudoku`. No upstream code is copied.

## Repository layout

Every example directory contains:

- `environment.py`: native Flash environment, reward, and deterministic task logic
- `data/train.jsonl`: reward-verified teacher trajectories
- `data/heldout.json`: 50 frozen cases not sent to the teacher
- one shipped `train.toml`, or ordered `train_sft.toml` plus `train_grpo.toml` or `train_opd.toml`
- `README.md`, `call.py`, and `smoke_test.py`
- `PROVENANCE.md` where an external environment inspired the task

Top-level tooling:

- [data-generation](data-generation): regenerate reward-verified teacher data from environment-native prompts and rewards
- [eval](eval): evaluate shipped adapters and GPT-5.5 on the same held-out rows
- [RESULTS.md](RESULTS.md): final campaign results, run ids, footprint, and ablations
- [VALIDATION.md](VALIDATION.md): local and live validation record

## Quickstart

Use Python 3.11 or 3.12.

```bash
uv sync
uv tool install --force freesolo-flash==0.2.58
flash login
flash whoami
uv run pytest
uv run ruff check .
```

## Train an example

Publish the example directory so the bundled `data/train.jsonl` is included with the environment. Replace the checked-in `[environment].id` if your new published id differs from the campaign id.

Single-stage recipe:

```bash
flash env push --name running-total-sft examples/running-total-sft
flash train examples/running-total-sft/train.toml --dry-run
flash train examples/running-total-sft/train.toml --cost
flash train examples/running-total-sft/train.toml --background
```

Two-stage warm start:

```bash
flash env push --name structured-number-guess-sft-grpo examples/structured-number-guess-sft-grpo
flash train examples/structured-number-guess-sft-grpo/train_sft.toml --background
```

After the SFT stage finishes, replace `init_from_adapter` in `train_grpo.toml` with the new parent run id, then validate cost and launch:

```bash
flash train examples/structured-number-guess-sft-grpo/train_grpo.toml --dry-run
flash train examples/structured-number-guess-sft-grpo/train_grpo.toml --cost
flash train examples/structured-number-guess-sft-grpo/train_grpo.toml --background
```

Warm-start child configs intentionally do not set `lora_rank` or `lora_alpha`; the adapter shape is inherited from the parent.

## Evaluate

> **Security warning:** Math Python evaluation executes model-generated Python directly on the host with no sandbox. It is disabled by default. Run it only in a disposable machine or container and explicitly pass `--allow-unsafe-local-code-execution`.

Evaluate a shipped adapter on all 50 frozen cases:

```bash
uv run python eval/evaluate_suite.py \
  --example running-total-sft \
  --output eval-results/running-total-model.json
```

Run the corresponding GPT-5.5 comparison through the local gateway:

```bash
uv run python eval/evaluate_gpt55.py \
  --example running-total-sft \
  --output eval-results/running-total-gpt55.json
```

See [eval/README.md](eval/README.md) for model, endpoint, and dry-run options.

## Regenerate distilled data

Set `OPENROUTER_API_KEY` in the process environment or pass a private env-file path outside the repository. Never commit the key.

```bash
export OPENROUTER_API_KEY="..."
uv run python data-generation/distill.py \
  --task math-boxed-sft \
  --output-dir generated/math-boxed-sft
```

See [data-generation/README.md](data-generation/README.md) for all eight tasks and multi-turn replay validation.

## Security and interpretation

- no secret values belong in this repository
- teacher credentials are read only at runtime
- held-out rows are never submitted to the teacher by the generation scripts
- math Python executes model-written code directly on the host with no sandbox; local evaluation requires explicit unsafe opt-in and a disposable machine or container
- provider-reported tokens use different tokenizers and hidden-reasoning conventions
- latency was measured on different serving stacks and is not a controlled comparison
- all completed results are task-specific comparisons, not a broad model ranking; number guess is parity, while logic and Sudoku are near parity under their corrected contracts
