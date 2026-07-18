# Verify Flash example runtime paths

Use the public evaluation CLI as the runtime surface. Do not use paid models, GPUs, or deployed adapters for local verification.

1. Start a local OpenAI-compatible stub endpoint that returns task-valid responses and records request bodies.
2. Run `uv run python eval/evaluate_suite.py` against the stub with `FREESOLO_API_KEY` set to a placeholder.
3. For `math-python-sft`, confirm the command refuses execution without `--allow-unsafe-local-code-execution`, permits `--dry-run` without the flag, and prints the host-execution warning when explicitly enabled.
4. For structured number guess, inspect recorded `response_format` schemas and confirm every case uses its held-out `low` and `high` bounds.
5. Probe strict reward paths with malformed logic answers and malformed Sudoku moves, then run valid episodes through completion.
6. Keep all outputs under `/tmp`; do not train, deploy, or call external model endpoints.
