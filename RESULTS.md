# Training campaign results

## Conclusion

The campaign produced eight 2.3B to 9.65B Qwen3.5 task specialists. All eight use teacher supervision through bundled SFT trajectories or OPD, and all eight have frozen 50-case evaluations under their active contracts.

Structured number guess and Math Python reach honest parity with GPT-5.5 under their corrected contracts. Logic boolean and Sudoku reach near parity under their corrected strict rewards. The old claim that number guess exceeded GPT-5.5 came from trivial constant-offset data and is retracted.

Some students score higher under a task's strict environment contract. Those differences are real for the frozen evaluations, but they mostly reflect exact formatting, tool-use, or episode-protocol slips by GPT-5.5 rather than broad capability superiority. The evidence supports shipping compact task specialists, not ranking the underlying models globally.

## Held-out quality

Each evaluation set contains 50 frozen held-out cases. Train and held-out inputs are disjoint, and held-out prompts were not sent to a teacher. The Math Python GPT-5.5 result reports the 46 cases that completed; four additional OpenRouter requests hung on provider latency and were not scored.

| Task                    | Base       | Shipped recipe      | Model reward or accuracy |            GPT-5.5 | Verdict                                   |
| ----------------------- | ---------- | ------------------- | -----------------------: | -----------------: | ----------------------------------------- |
| Running total           | Qwen3.5-2B | pure SFT            |                     100% |               100% | parity                                    |
| Logic boolean           | Qwen3.5-4B | SFT to GRPO         |       48/50, reward 0.96 |              50/50 | near parity, two cases behind             |
| Structured number guess | Qwen3.5-2B | SFT to GRPO         |   50/50, mean reward 1.0 |    50/50, mean 1.0 | parity on genuine varied-secret cases     |
| Thinking science        | Qwen3.5-9B | single-stage OPD    |                    50/50 |              50/50 | parity                                    |
| Math boxed              | Qwen3.5-9B | pure SFT            |            0.90 accuracy |      0.92 accuracy | near parity, one of 50 cases behind       |
| Math Python             | Qwen3.5-4B | pure SFT            |             49/50, 98.0% |       45/46, 97.8% | parity under enforced tool use            |
| Thinking math           | Qwen3.5-9B | SFT to OPD          |            0.92 accuracy |      0.92 accuracy | parity                                    |
| Sudoku                  | Qwen3.5-4B | SFT to GRPO, step 8 |       49/50, mean 5.5714 | 50/50, mean 5.6700 | near parity under strict corrected reward |

The original campaign GPT-5.5 measurements used the local gateway. Math Python was re-measured through OpenRouter because the gateway was unreliable and its timeouts had depressed the original reported result to 82%. Under the corrected contract, a boxed answer receives credit only after an executed Python tool turn; the adapter scored 49/50 and GPT-5.5 scored 45/46 completed OpenRouter cases, both approximately 98%.

Six recipes use Kimi K2.6 (`moonshotai/kimi-k2.6`) supervision, and logic boolean plus Sudoku use GLM-5.2 (`z-ai/glm-5.2`) trajectories. Logic boolean's strict-normalized GLM-5.2 corpus is consumed by its SFT warm start.

## Shipped runs

| Task                    | SFT run                     | RL or OPD run               | Shipped selector |
| ----------------------- | --------------------------- | --------------------------- | ---------------- |
| Running total           | `flash-1784320041-9c4a32b8` | none                        | final / step 100 |
| Logic boolean           | `flash-1784350210-5225a6a4` | `flash-1784350975-6dc07970` | final            |
| Structured number guess | `flash-1784349216-ba05be67` | `flash-1784349640-ceb35b18` | final            |
| Thinking science        | none                        | `flash-1784321193-dab97aef` | final            |
| Math boxed              | `flash-1784263689-f98515ce` | none                        | final / step 75  |
| Math Python             | `flash-1784322317-e152ffdd` | none                        | final / step 75  |
| Thinking math           | `flash-1784325488-7a1d31b9` | `flash-1784326094-ab33c65b` | final            |
| Sudoku                  | `flash-1784324917-1d8f0ce8` | `flash-1784327754-e1d3a602` | step 8           |

Warm-start child configs intentionally omit `lora_rank` and `lora_alpha`. The child adapter inherits the parent LoRA shape through `init_from_adapter`.

## Parameter and operational footprint

The students are small open checkpoints: approximately 2.3B parameters for Qwen3.5-2B, 4.66B for Qwen3.5-4B, and 9.65B for Qwen3.5-9B. GPT-5.5's parameter count is not public, so this repository does not publish a numeric parameter ratio or imply a known architecture-level comparison.

Provider-reported mean total tokens per evaluated case or episode were retained for the original four unaffected evaluations. The corrected Phase 2 artifacts for logic, number guess, and Sudoku and the enforced-contract Math Python re-evaluation did not retain comparable token usage, so no token number is inferred for those rows.

| Task                    | Student parameters |                      Student tokens |                      GPT-5.5 tokens | Observation                                                               |
| ----------------------- | -----------------: | ----------------------------------: | ----------------------------------: | ------------------------------------------------------------------------- |
| Running total           |               2.3B |                   75.50 per request |                  386.63 per request | student used fewer reported tokens                                        |
| Logic boolean           |              4.66B | not collected in final Phase 2 eval | not collected in final Phase 2 eval | both evaluations used 50 single-turn cases                                |
| Structured number guess |               2.3B | not collected in final Phase 2 eval | not collected in final Phase 2 eval | 292 student turns versus 289 GPT-5.5 turns across 50 solved cases         |
| Thinking science        |              9.65B |                     285.28 per case |                     445.16 per case | student used fewer reported tokens                                        |
| Math boxed              |              9.65B |                     288.78 per case |                     599.14 per case | student used fewer reported tokens                                        |
| Math Python             |              4.66B |   not collected in enforced re-eval |   not collected in enforced re-eval | both results enforce an executed Python tool turn before the boxed answer |
| Thinking math           |              9.65B |                     247.50 per case |                     567.86 per case | student used fewer reported tokens                                        |
| Sudoku                  |              4.66B | not collected in final Phase 2 eval | not collected in final Phase 2 eval | 503 student turns versus 495 GPT-5.5 turns across 50 cases                |

Token accounting is provider-specific and can include different hidden-reasoning conventions and tokenizers. It is useful as an operational footprint signal, not a perfectly controlled model comparison. Episode turns are protocol counts, not substitutes for tokens.

Latency is deliberately not presented as a campaign win. Student adapters were measured on Flash's shared Modal serving path; the original GPT-5.5 campaign used a local gateway, while the corrected Math Python comparison used OpenRouter. These paths have different concurrency, retries, caching, and infrastructure. Running total happened to be about 6.8 times faster in the observed setup, but that is not a controlled latency benchmark and should not be generalized.

## SFT versus RL ablations

The ablations support a bounded conclusion: SFT establishes most of the task behavior, while post-SFT RL can help when a strict environment still exposes a learnable failure mode. RL is not uniformly beneficial.

- Logic boolean: strict-normalized SFT scored 45/50. Warm-start GRPO raised the result to 48/50. Single-stage GRPO scored 0/50 because strict all-or-nothing reward gave zero reward to cold-start rollouts that missed the exact output contract. A 9B SFT-to-GRPO run also scored 48/50 on the same two deeply nested `True` cases, so 4B shipped.
- Structured number guess: the corrected SFT-to-GRPO adapter reached 50/50 and mean reward 1.0, matching GPT-5.5. No corrected frozen SFT-only evaluation was retained, so the final result does not attribute the parity outcome specifically to GRPO.
- Math boxed: 9B SFT scored 0.88 in the detailed ablation, GRPO fell to 0.84, and OPD tied SFT at 0.88. The final campaign comparison uses the sealed 0.90 accuracy result.
- Thinking math: 9B SFT scored 0.94; OPD reduced it to 0.92. OPD is shipped to demonstrate the warm-start workflow, not because it improved quality.
- Sudoku: the selected step-8 SFT-to-GRPO adapter scored 49/50 and 5.5714 under the strict corrected reward. The earlier SFT-versus-GRPO numbers predated the corrected reward and remain superseded, so no updated causal improvement claim is made.
- Math Python: multi-turn OPD was too slow to reach a deployable checkpoint and was cancelled; pure SFT shipped. The enforced-tool-use re-evaluation places the adapter and GPT-5.5 at parity, approximately 98%.
- Tiny-base OPD and RL attempts were unstable or below parity on several tasks, so the final recipes use the smallest tested student that met the held-out gate rather than forcing one algorithm everywhere.

The practical pattern is to establish the exact output or episode contract with verified SFT data, add RL only when the environment exposes a measurable remaining failure mode, and retain the simpler SFT adapter when RL does not improve the frozen held-out set.

## Reproducibility boundaries

- `examples/*/data/train.jsonl` contains reward-verified teacher trajectories. Logic boolean's file is strict-normalized and consumed by `train_sft.toml`.
- `examples/*/data/heldout.json` contains the frozen 50-case comparison split.
- [data-generation](data-generation) regenerates the data from environment-native prompts and rewards.
- `flash env eval <run-id>` grades a deployed adapter against the frozen split, using each example's `evaluations.py` sidecar and the environment's own reward.
- The GPT-5.5 numbers below were produced by a standalone evaluator that has since been deleted along with the rest of `eval/`. They are reported as recorded and are not reproducible from this repository as it stands; reproducing them needs an evaluator that can send the held-out cases to an arbitrary external model.
- Exact training configs and warm-start run ids are checked into each example directory, except that the number-guess config retains its original campaign parent id and must be updated to a newly reproduced SFT run before launching its GRPO child.

All eight results are task-specific. None establishes broad benchmark dominance, controlled latency superiority, or a known parameter ratio to GPT-5.5.
