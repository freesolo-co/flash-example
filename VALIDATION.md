# Validation

This repository consolidates an eight-task training campaign. Recipe coverage is three pure-SFT examples, one single-stage OPD example, three SFT-to-GRPO examples, and one SFT-to-OPD example. There is no pure single-stage-GRPO example: logic boolean cold-started at 0/50 under its strict all-or-nothing reward and now uses an SFT warm start.

## Local repository validation

The final assembled tree is validated with:

```bash
uv run pytest
uv run ruff check .
```

The tests cover:

- all eight environment modules loading deterministic datasets
- exact bundled train row counts and 50 held-out rows per task
- train and held-out input disjointness
- shipped model, algorithm, step budget, and environment ids for every config
- warm-start parent run ids and the absence of child `lora_rank` or `lora_alpha`
- strict parsers and native reward contracts
- strict normalization of all 150 logic SFT rows
- record-metadata fallback for reproducible number-guess SFT episodes
- complete multi-turn lifecycle checks for running total, number guess, math Python, and Sudoku

The evaluation and data-generation command-line tools are also exercised in network-free help or dry-run mode during final validation.

## Frozen data

| Task                    | Verified train rows | Held-out rows | Teacher                            |
| ----------------------- | ------------------: | ------------: | ---------------------------------- |
| Running total           |                 100 |            50 | Kimi K2.6                          |
| Logic boolean           |                 150 |            50 | GLM-5.2, strict-normalized for SFT |
| Structured number guess |                 100 |            50 | Kimi K2.6                          |
| Thinking science        |                 135 |            50 | Kimi K2.6                          |
| Math boxed              |                 143 |            50 | Kimi K2.6                          |
| Math Python             |                  97 |            50 | Kimi K2.6                          |
| Thinking math           |                 143 |            50 | Kimi K2.6                          |
| Sudoku                  |                  99 |            50 | GLM-5.2                            |

Generation used reward verification and rejection sampling. Held-out rows were created before teacher execution, are disjoint from the retained train inputs, and were not included in teacher-attempt logs.

## Shipped live runs

| Task                    | Recipe      | Live run ids                                                    | Held-out model result |            GPT-5.5 |
| ----------------------- | ----------- | --------------------------------------------------------------- | --------------------: | -----------------: |
| Running total           | pure SFT    | `flash-1784320041-9c4a32b8`                                     |                  100% |               100% |
| Logic boolean           | SFT to GRPO | `flash-1784350210-5225a6a4`, `flash-1784350975-6dc07970`        |      48/50, 0.96 mean |              50/50 |
| Structured number guess | SFT to GRPO | `flash-1784349216-ba05be67`, `flash-1784349640-ceb35b18`        |       50/50, 1.0 mean |    50/50, 1.0 mean |
| Thinking science        | OPD         | `flash-1784321193-dab97aef`                                     |                 50/50 |              50/50 |
| Math boxed              | pure SFT    | `flash-1784263689-f98515ce`                                     |                  0.90 |               0.92 |
| Math Python             | pure SFT    | `flash-1784322317-e152ffdd`                                     |                   92% |                82% |
| Thinking math           | SFT to OPD  | `flash-1784325488-7a1d31b9`, `flash-1784326094-ab33c65b`        |                  0.92 |               0.92 |
| Sudoku                  | SFT to GRPO | `flash-1784324917-1d8f0ce8`, `flash-1784327754-e1d3a602` step 8 |    49/50, 5.5714 mean | 50/50, 5.6700 mean |

These were paid real-GPU Flash runs on RunPod RTX 4090, A100 PCIe, or H100 hardware. Hardware is recorded in campaign artifacts and in the checked-in configs that include a `[gpu]` section; not every shipped config records it.

All eight adapters were evaluated against each task's 50 frozen cases. The corrected Phase 2 evaluations establish number-guess parity, logic near parity by two cases, and Sudoku near parity by one case under the strict corrected reward.

## Interpretation checks

The documentation intentionally preserves these boundaries:

- all eight results are task-specific comparisons, not a general superiority claim
- strict-contract wins are reported as task-specific exceeds, not broad model wins
- number guess is parity on genuine varied-secret data; the old constant-offset exceeds claim is retracted
- logic and Sudoku are near parity, not parity or superiority
- GPT-5.5's parameter count is undisclosed, so no numeric parameter ratio is claimed
- provider token counts are reported as operational footprint signals, not perfectly comparable tokenization measurements
- the corrected Phase 2 artifacts did not retain token usage, so no token values are inferred for logic, number guess, or Sudoku
- latency is not presented as controlled because the serving stacks and concurrency differ
- thinking-math OPD regressed from 0.94 SFT to 0.92 and is retained only to demonstrate the shipped warm-start workflow
- multi-turn math Python OPD was cancelled for throughput before a deployable checkpoint, so pure SFT shipped

See [RESULTS.md](RESULTS.md) for the complete results and ablation record.

## Security validation

Before committing, the repository is scanned for:

- the exact private OpenRouter key value from the external campaign env file
- common private-key and credential assignment patterns
- accidental copies of `.env` files

Only environment-variable names and placeholder values may appear in documentation. The private env file remains outside the repository.
