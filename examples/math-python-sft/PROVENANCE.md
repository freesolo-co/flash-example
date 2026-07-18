# Provenance

## Source

- Prime Intellect environments hub: `primeintellect/math-python`
- version: latest, pulled 2026-07-16
- upstream GitHub: <https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/math_python>

## Reproduced contract

This port reproduces the tool-use math task, the requirement that the model use Python during a multi-turn solution loop, the final `\\boxed{}` answer-format contract, and symbolic-style answer verification. Numeric integers, decimals, and rational `a/b` forms are compared as exact fractions; other answers use normalized string comparison.

## Self-contained adaptation

The source uses the `verifiers` framework, an externally loaded example math dataset, and a remote Prime sandbox through `vf.PythonEnv`. This port replaces those dependencies with a deterministic 24-problem in-folder dataset, the `freesolo.environments` multi-turn API, and a bounded local subprocess executor. The local executor has a five-second timeout, a fresh temporary working directory, truncated output, and a child environment containing only `PATH` and `PYTHONIOENCODING`.

The bounded subprocess is a self-containment mechanism, not a security boundary. Use a real isolated sandbox for untrusted code, larger models, or production workloads.

No source code was copied verbatim. The task and verification logic were reimplemented for the current Freesolo API.

## Limitations

The fixed 24-problem pool is intentionally small and bounded. It is an instructional training example, not a comprehensive math benchmark. Its exact-fraction and normalized-string comparison is narrower than a full computer algebra verifier, and local subprocess execution inherits the host user's filesystem and operating-system permissions.
