# Provenance

The bundled science questions, answer choices, and answer keys are externally sourced from [AllenAI's OpenBookQA dataset](https://allenai.org/data/open-book-qa). `data/train.jsonl` uses a deterministic sample of the OpenBookQA `Additional` training split with reward-verified Kimi K2.6 completions. `data/heldout.json` uses a disjoint deterministic sample of the OpenBookQA `Additional` test split that was not sent to the teacher. The held-out IDs and `source_split` metadata preserve that origin.

Reuse of the bundled data must retain this OpenBookQA attribution and comply with the upstream dataset terms.

The repository organization is inspired by the [OpenPipe ART examples](https://github.com/OpenPipe/ART/tree/main/examples), but no ART code was copied.
