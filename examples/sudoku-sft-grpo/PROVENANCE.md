# Provenance

## Source

This example is based on Prime Intellect Environments Hub environment `m8ngotree/sudoku`, version `latest`, pulled 2026-07-16 from the Prime Intellect Environments Hub. No upstream GitHub URL was present in the supplied source materials.

## Reproduced behavior

The port reproduces the one-move-per-turn Sudoku task, the `<move>A1=5</move>` action protocol including `0` for clearing a cell, row and column coordinates, board constraint checks, deterministic seeded puzzle generation, updated-board and progress feedback, and the shaped final reward with solved and efficiency bonuses, partial progress credit, and constraint-violation penalties.

Each generated example stores its puzzle and unique solution in metadata. The configured easy pool uses 8 to 12 blanks so a complete episode remains bounded.

## Self-contained adaptation

The source uses the `verifiers` framework and the `datasets` package. This port replaces those dependencies with inlined deterministic generator, solver, parser, board, feedback, and reward logic implemented against the `freesolo.environments` multi-turn API. It uses no remote dataset, sandbox, network call, or secret during environment construction and scoring.

No source code was copied verbatim. The behavior and algorithms were reimplemented for this self-contained Freesolo example.

## Limitations

The deterministic 24-example near-complete pool is intended for bounded Flash examples and smoke tests. It is not a representative Sudoku benchmark, and its puzzles are substantially easier and shorter than the upstream default easy generation range.
