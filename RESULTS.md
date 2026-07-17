# Distillation campaign results

## Conclusion

The campaign distilled strong teacher behavior into 2.3B to 9.65B Qwen3.5 students that **match GPT-5.5's per-task quality** across all eight shipped examples. This is a match claim, not a general claim that the students beat GPT-5.5.

Several students score higher under a task's strict environment contract. Those differences are real for these frozen evaluations, but they mostly reflect exact formatting, tool-use, or episode-protocol slips by GPT-5.5 rather than broad capability superiority. The evidence supports shipping compact task specialists, not ranking the underlying models globally.

## Held-out quality

Each row uses 50 frozen held-out cases. Train and held-out inputs are disjoint, and held-out prompts were not sent to the distillation teacher.

| Task                    | Base       | Shipped recipe      | Model reward or accuracy |             GPT-5.5 | Verdict                               |
| ----------------------- | ---------- | ------------------- | -----------------------: | ------------------: | ------------------------------------- |
| Running total           | Qwen3.5-2B | pure SFT            |                     100% |                100% | match                                 |
| Logic boolean           | Qwen3.5-4B | single-stage GRPO   |                    50/50 |               50/50 | match                                 |
| Structured number guess | Qwen3.5-2B | SFT to GRPO         |              100% solved |          88% solved | exceeds this strict contract          |
| Thinking science        | Qwen3.5-9B | single-stage OPD    |                    50/50 |               50/50 | match                                 |
| Math boxed              | Qwen3.5-9B | pure SFT            |            0.90 accuracy |       0.92 accuracy | match within the campaign parity band |
| Math Python             | Qwen3.5-4B | pure SFT            |      92% reward accuracy | 82% reward accuracy | exceeds this strict contract          |
| Thinking math           | Qwen3.5-9B | SFT to OPD          |            0.92 accuracy |       0.92 accuracy | match                                 |
| Sudoku                  | Qwen3.5-4B | SFT to GRPO, step 8 |              100% solved |         100% solved | match                                 |

The teachers were Kimi K2.6 (`moonshotai/kimi-k2.6`) for six tasks. Logic boolean and Sudoku used GLM-5.2 (`z-ai/glm-5.2`) because Kimi K2.6 had lower reward-verified yield under those tasks' strict terminal protocols.

## Shipped runs

| Task                    | SFT run                     | RL or OPD run               | Shipped selector |
| ----------------------- | --------------------------- | --------------------------- | ---------------- |
| Running total           | `flash-1784320041-9c4a32b8` | none                        | final / step 100 |
| Logic boolean           | none                        | `flash-1784316952-a904a84d` | final            |
| Structured number guess | `flash-1784319868-8f8ee7be` | `flash-1784321381-dbe68392` | final            |
| Thinking science        | none                        | `flash-1784321193-dab97aef` | final            |
| Math boxed              | `flash-1784263689-f98515ce` | none                        | final / step 75  |
| Math Python             | `flash-1784322317-e152ffdd` | none                        | final / step 75  |
| Thinking math           | `flash-1784325488-7a1d31b9` | `flash-1784326094-ab33c65b` | final            |
| Sudoku                  | `flash-1784324917-1d8f0ce8` | `flash-1784327754-e1d3a602` | step 8           |

Warm-start child configs intentionally omit `lora_rank`. The child adapter inherits the parent LoRA shape through `init_from_adapter`.

## Parameter and token footprint

The students are small open checkpoints: approximately 2.3B parameters for Qwen3.5-2B, 4.66B for Qwen3.5-4B, and 9.65B for Qwen3.5-9B. GPT-5.5's parameter count is not public, so this repository does not publish a numeric parameter ratio or imply a known architecture-level comparison.

Provider-reported mean total tokens per evaluated case or episode were:

| Task                    | Student parameters |        Student tokens |                 GPT-5.5 tokens | Observation                                                                  |
| ----------------------- | -----------------: | --------------------: | -----------------------------: | ---------------------------------------------------------------------------- |
| Running total           |               2.3B |     75.50 per request |             386.63 per request | student used fewer reported tokens                                           |
| Logic boolean           |              4.66B |       378.42 per case |                519.10 per case | student used fewer reported tokens                                           |
| Structured number guess |               2.3B |  1,449.00 per episode |           3,288.74 per episode | student used fewer reported tokens                                           |
| Thinking science        |              9.65B |       285.28 per case |                445.16 per case | student used fewer reported tokens                                           |
| Math boxed              |              9.65B |       288.78 per case |                599.14 per case | student used fewer reported tokens                                           |
| Math Python             |              4.66B |     1,865.60 per case | 4,694.74 per observed GPT case | student used fewer reported tokens; GPT token mean covers 47 completed cases |
| Thinking math           |              9.65B |       247.50 per case |                567.86 per case | student used fewer reported tokens                                           |
| Sudoku                  |              4.66B | 26,014.58 per episode |          23,610.08 per episode | student used more reported tokens                                            |

Token accounting is provider-specific and can include different hidden-reasoning conventions and tokenizers. It is useful as an operational footprint signal, not a perfectly controlled model comparison.

Latency is deliberately not presented as a campaign win. Student adapters were measured on Flash's shared Modal serving path, while GPT-5.5 was measured through a local gateway with different concurrency, retries, caching, and infrastructure. Running total happened to be about 6.8 times faster in the observed setup, but that is not a controlled latency benchmark and should not be generalized.

## SFT versus RL ablations

The completed ablations favor a simple conclusion: distilled SFT carried most of the quality, while post-SFT RL usually tied or regressed.

- Structured number guess: SFT already solved 100%; GRPO sampled all-reward-1 groups with zero variance and made no effective update.
- Math boxed: 9B SFT scored 0.88 in the detailed ablation, GRPO fell to 0.84, and OPD tied SFT at 0.88. The final campaign comparison uses the sealed 0.90 accuracy result.
- Thinking math: 9B SFT scored 0.94; OPD reduced it to 0.92. OPD is shipped to demonstrate the warm-start workflow, not because it improved quality.
- Sudoku: GRPO was the clear positive case, improving SFT from 94% solved to 100% solved at step 8.
- Math Python: multi-turn OPD was too slow to reach a deployable checkpoint and was cancelled; pure SFT shipped.
- Tiny-base OPD and RL attempts were unstable or below parity on several tasks, so the final recipes use the smallest tested student that met the held-out gate rather than forcing one algorithm everywhere.

The practical recipe is therefore: distill verified trajectories with SFT first, add RL only when the environment exposes a measurable remaining failure mode, and keep the SFT adapter when RL does not improve the frozen held-out set.

## Reproducibility boundaries

- `examples/*/data/train.jsonl` contains the reward-verified teacher trajectories.
- `examples/*/data/heldout.json` contains the frozen 50-case comparison split.
- [data-generation](data-generation) regenerates the data from environment-native prompts and rewards.
- [eval](eval) evaluates shipped adapters and GPT-5.5 through the same environment lifecycle.
- Exact training configs and warm-start run ids are checked into each example directory.

These results establish eight task-specific matches under frozen contracts. They do not establish broad benchmark dominance, controlled latency superiority, or a known parameter ratio to GPT-5.5.
