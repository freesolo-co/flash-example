#!/usr/bin/env python3
"""replay and validate generated multi-turn distillation artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from distill import load_env_file
from distill_multiturn import ADAPTERS, assert_disjoint, problem_digest
from freesolo.environments import EnvironmentEpisode


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay_row(adapter: Any, problem: Any, row: dict[str, Any]) -> None:
    transcript = list(row["output"]["messages"])
    example = adapter.task_example(problem)
    messages = list(adapter.environment.start_episode(example, problem.input))
    index = 0
    done = False
    last_response = ""
    while index < len(transcript):
        assistant = transcript[index]
        if assistant.get("role") != "assistant":
            raise ValueError(f"{problem.id}: expected assistant at transcript index {index}")
        last_response = str(assistant["content"])
        before_response = list(messages)
        step = adapter.call_step(example, before_response, last_response)
        messages.append({"role": "assistant", "content": last_response})
        index += 1
        expected_feedback = [dict(message) for message in step.messages]
        actual_feedback = transcript[index : index + len(expected_feedback)]
        if actual_feedback != expected_feedback:
            raise ValueError(f"{problem.id}: environment feedback mismatch")
        messages.extend(actual_feedback)
        index += len(expected_feedback)
        done = bool(step.done)
        if done and index != len(transcript):
            raise ValueError(f"{problem.id}: transcript continues after terminal step")
    if not done:
        raise ValueError(f"{problem.id}: transcript did not reach a terminal step")
    episode = EnvironmentEpisode(messages=tuple(messages), response_text=last_response)
    reward = adapter.environment.score_episode(example, episode)
    valid, reason = adapter.validate_protocol(problem, transcript, reward)
    if not reward.success or not valid:
        raise ValueError(f"{problem.id}: replay failed: {reason}")


def assert_no_secret(output_dir: Path, env_file: Path | None) -> None:
    secret = os.environ.get("OPENROUTER_API_KEY") or load_env_file(env_file).get(
        "OPENROUTER_API_KEY"
    )
    if not secret:
        return
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        if secret in content:
            raise ValueError(f"secret value found in generated file: {path}")


def validate(task: str, output_dir: Path, env_file: Path | None) -> dict[str, Any]:
    adapter = ADAPTERS[task]()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    train_size = int(manifest["split"]["requested_train"])
    heldout_size = int(manifest["split"]["heldout"])
    seed = int(manifest["split"]["base_seed"])
    train_problems, heldout_problems = adapter.build_splits(
        train_size, heldout_size, seed
    )
    assert_disjoint(train_problems, heldout_problems)
    if len({problem_digest(problem) for problem in train_problems}) != len(train_problems):
        raise ValueError("train problems are not unique")
    if len({problem_digest(problem) for problem in heldout_problems}) != len(
        heldout_problems
    ):
        raise ValueError("held-out problems are not unique")

    by_input = {problem.input: problem for problem in train_problems}
    if len(by_input) != len(train_problems):
        raise ValueError("train inputs are not unique")
    train_rows = read_jsonl(output_dir / "train.jsonl")
    if len(train_rows) != int(manifest["split"]["kept_train"]):
        raise ValueError("train row count differs from manifest")
    for row in train_rows:
        problem = by_input.get(str(row["input"]))
        if problem is None:
            raise ValueError("train row does not belong to the generated train split")
        replay_row(adapter, problem, row)

    heldout_rows = json.loads((output_dir / "heldout.json").read_text())
    if len(heldout_rows) != heldout_size:
        raise ValueError("held-out row count differs from manifest")
    if {row["id"] for row in heldout_rows} != {
        problem.id for problem in heldout_problems
    }:
        raise ValueError("held-out ids differ from the generated held-out split")

    attempts = read_jsonl(output_dir / "attempts.jsonl")
    train_ids = {problem.id for problem in train_problems}
    heldout_ids = {problem.id for problem in heldout_problems}
    attempt_ids = {str(row["id"]) for row in attempts}
    if len(attempts) != train_size or attempt_ids != train_ids:
        raise ValueError("attempt records do not cover the train split exactly")
    if attempt_ids & heldout_ids:
        raise ValueError("held-out ids appear in teacher attempt records")
    if int(manifest["split"]["heldout_teacher_requests"]) != 0:
        raise ValueError("manifest does not record zero held-out teacher requests")

    assert_no_secret(output_dir, env_file)
    return {
        "task": task,
        "verified_train": len(train_rows),
        "heldout": len(heldout_rows),
        "overlap": 0,
        "heldout_teacher_requests": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-env-file", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate(args.task, args.output_dir, args.api_env_file)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
