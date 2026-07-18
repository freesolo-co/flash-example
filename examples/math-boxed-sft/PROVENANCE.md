# Provenance

- Source: Prime Intellect environments hub environment `primeintellect/math-env`, version `latest`, pulled 2026-07-16.
- Upstream GitHub: [`PrimeIntellect-ai/verifiers/environments/math_env`](https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/math_env).
- Reproduced behavior: the single-turn math task, the instruction to explain reasoning and place the final answer in `\boxed{}`, strict boxed-answer extraction, and the intent of rule-based symbolic equivalence checking with the optional judge disabled.
- Self-contained adaptation: the source defaults to the Hugging Face `PrimeIntellect/INTELLECT-3-RL` dataset with subset `math` and depends on `verifiers`, `math_verify`, and SymPy-backed verification. This port replaces those dependencies with 24 deterministic, original in-process math problems and a reimplemented normalized checker using only standard-library rational arithmetic through `fractions.Fraction` and the `freesolo.environments` API.
- No source code was copied verbatim. The task and verification logic were reimplemented for this example.

The bounded problem pool is intended for example training and smoke validation, not as a benchmark. The checker recognizes integers, decimals, slash fractions, simple numeric LaTeX `\frac{a}{b}` forms, and normalized string equality. It does not provide general symbolic algebra equivalence or an LLM judge fallback.
