# Provenance

## Source

- Prime Intellect environments hub environment: `primeintellect/logic-env`
- Version: latest, pulled 2026-07-16
- Upstream GitHub: [`PrimeIntellect-ai/verifiers/environments/logic_env`](https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/logic_env)

## What this port reproduces

This port reproduces the single `boolean_expressions` task as a representative task from the source's 40-plus-task logic corpus. It preserves the task intent of evaluating nested boolean expressions, the contract that the model may reason and must provide its final value inside `<answer>...</answer>`, and logic_env's exact-match verification of the extracted answer against the gold value.

The generated expressions use only `True`, `False`, `and`, `or`, `not`, and parentheses. Truth values are computed structurally while each expression tree is generated rather than by evaluating arbitrary input.

## Self-contained adaptation

The source environment loads the Hugging Face `PrimeIntellect/INTELLECT-3-RL` dataset with the `logic` subset and dispatches across a large verifier framework covering more than 40 tasks. Some source tasks also require large word-list dependencies. This port does not import that dataset, the `verifiers` framework, or any word lists. It replaces them with a small deterministic in-process boolean-expression generator and reimplements the answer extraction and exact-match reward through the `freesolo.environments` API.

No source code was copied verbatim. The task behavior and verification contract were reimplemented for this self-contained example.

## Limitations

The GRPO environment uses a bounded pool of 24 synthetic expressions with controlled depths from 1 through 5, while the SFT warm start uses 150 disjoint teacher-completed expressions. The example covers only boolean expressions, not the rest of the source logic corpus. It is intended as a compact training example, not as a benchmark or a reproduction of the source dataset distribution. Exact-match reward checks only the final tagged truth value and does not grade the reasoning trace.
