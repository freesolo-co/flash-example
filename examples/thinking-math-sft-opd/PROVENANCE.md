# Provenance

This example adapts the scoring contract from an internal Freesolo GSM8K environment.

The bundled problems and canonical answers are externally sourced from [OpenAI's GSM8K dataset](https://github.com/openai/grade-school-math). `data/train.jsonl` uses a deterministic sample of the official GSM8K training split with reward-verified Kimi K2.6 completions. `data/heldout.json` uses a disjoint deterministic sample of the official GSM8K test split that was not sent to the teacher. The held-out IDs and `source_split` metadata preserve that origin.

Reuse of the bundled data must retain this GSM8K attribution and comply with the upstream dataset terms.
