#!/usr/bin/env python3
"""generate reward-verified multi-turn sft episodes with an openrouter teacher."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from distill import OPENROUTER_ENDPOINT, OpenRouterTeacher, Usage, load_env_file, write_json
from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEpisode

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "examples"
TASK_MODULES = {
    "running-total-sft": ROOT / "running-total-sft" / "environment.py",
    "structured-number-guess-grpo": ROOT
    / "structured-number-guess-grpo"
    / "environment.py",
    "math-python-grpo": ROOT / "math-python-grpo" / "environment.py",
    "sudoku-grpo": ROOT / "sudoku-grpo" / "environment.py",
}
TEACHERS = {
    "running-total-sft": "moonshotai/kimi-k2.6",
    "structured-number-guess-grpo": "moonshotai/kimi-k2.6",
    "math-python-grpo": "moonshotai/kimi-k2.6",
    "sudoku-grpo": "z-ai/glm-5.2",
}
DEFAULT_MAX_TOKENS = {
    "running-total-sft": 64,
    "structured-number-guess-grpo": 64,
    "math-python-grpo": 384,
    "sudoku-grpo": 1536,
}


@dataclass(frozen=True)
class EpisodeProblem:
    id: str
    row: dict[str, Any]
    split: str
    seed: str

    @property
    def input(self) -> str:
        return str(self.row["input"])


@dataclass
class EpisodeAttempt:
    kept: bool
    transcript: list[dict[str, str]]
    reason: str
    reward_score: float
    turns: int
    usage: Usage


@dataclass
class EpisodeResult:
    problem: EpisodeProblem
    kept: bool
    transcript: list[dict[str, str]] | None
    attempts: int
    reasons: list[str]
    reward_score: float
    turns: int
    usage: Usage
    attempt_records: list[dict[str, Any]] = field(default_factory=list)


class EpisodeAdapter(Protocol):
    task_name: str
    environment_path: Path
    source_description: dict[str, Any]

    def build_splits(
        self, train_size: int, heldout_size: int, seed: int
    ) -> tuple[list[EpisodeProblem], list[EpisodeProblem]]: ...

    def drive(
        self, problem: EpisodeProblem, teacher: OpenRouterTeacher
    ) -> EpisodeAttempt: ...

    def heldout_record(self, problem: EpisodeProblem) -> dict[str, Any]: ...


class ImportedMultiTurnAdapter:
    task_name = ""
    environment_path = Path()
    source_description: dict[str, Any] = {}

    def __init__(self) -> None:
        module_name = "flash_distill_" + self.task_name.replace("-", "_")
        spec = importlib.util.spec_from_file_location(module_name, self.environment_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not import environment module: {self.environment_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.module = module
        self.environment = self.load_environment()

    def load_environment(self) -> Any:
        return self.module.load_environment()

    @staticmethod
    def task_example(problem: EpisodeProblem) -> TaskExample:
        row = problem.row
        return TaskExample(
            record=row,
            id=str(row.get("id") or problem.id),
            input=str(row["input"]),
            output=row.get("output"),
            metadata=dict(row.get("metadata") or {}),
        )

    def call_step(
        self,
        example: TaskExample,
        messages_before_response: list[dict[str, str]],
        assistant_response: str,
    ) -> Any:
        messages_with_response = [
            *messages_before_response,
            {"role": "assistant", "content": assistant_response},
        ]
        return self.environment.step_episode(
            example, messages_with_response, assistant_response
        )

    def validate_protocol(
        self,
        problem: EpisodeProblem,
        transcript: list[dict[str, str]],
        reward: Any,
    ) -> tuple[bool, str]:
        _ = problem, transcript
        if not bool(reward.success):
            return False, str(reward.reason)
        return True, str(reward.reason)

    def complete_teacher(
        self,
        messages: list[dict[str, str]],
        teacher: OpenRouterTeacher,
    ) -> tuple[str, dict[str, Any]]:
        return teacher.complete(messages)

    def drive(
        self, problem: EpisodeProblem, teacher: OpenRouterTeacher
    ) -> EpisodeAttempt:
        example = self.task_example(problem)
        messages = list(self.environment.start_episode(example, problem.input))
        opening_length = len(messages)
        usage = Usage()
        last_response = ""
        step_done = False
        for _ in range(self.environment.max_episode_turns(example)):
            response, request_usage = self.complete_teacher(messages, teacher)
            usage.add(request_usage)
            last_response = response
            messages_before_response = list(messages)
            step = self.call_step(example, messages_before_response, response)
            messages.append({"role": "assistant", "content": response})
            messages.extend(dict(message) for message in step.messages)
            step_done = bool(step.done)
            if step_done:
                break

        transcript = [dict(message) for message in messages[opening_length:]]
        episode = EnvironmentEpisode(
            messages=tuple(messages), response_text=last_response
        )
        reward = self.environment.score_episode(example, episode)
        valid, protocol_reason = self.validate_protocol(problem, transcript, reward)
        kept = step_done and bool(reward.success) and valid
        if not step_done:
            reason = f"turn limit reached; {reward.reason}"
        elif not reward.success:
            reason = str(reward.reason)
        else:
            reason = protocol_reason
        turns = sum(message["role"] == "assistant" for message in transcript)
        return EpisodeAttempt(
            kept=kept,
            transcript=transcript,
            reason=reason,
            reward_score=float(reward.score),
            turns=turns,
            usage=usage,
        )

    def heldout_record(self, problem: EpisodeProblem) -> dict[str, Any]:
        row = problem.row
        return {
            "id": problem.id,
            "input": row["input"],
            "expected": row.get("output"),
            "metadata": dict(row.get("metadata") or {}),
            "generator_seed": problem.seed,
        }


class RunningTotalAdapter(ImportedMultiTurnAdapter):
    task_name = "running-total-sft"
    environment_path = TASK_MODULES[task_name]
    source_description = {
        "generator": "environment.build_dataset",
        "split_scheme": (
            "train uses the requested base seed; held-out uses base seed + 1000003; "
            "input digests are checked for disjointness"
        ),
    }

    def load_environment(self) -> Any:
        return self.module.load_environment(num_examples=1, seed=0)

    def build_splits(
        self, train_size: int, heldout_size: int, seed: int
    ) -> tuple[list[EpisodeProblem], list[EpisodeProblem]]:
        heldout_seed = seed + 1_000_003
        train_candidates = self.module.build_dataset(train_size * 10, seed)
        train_rows = select_unique_rows(train_candidates, train_size)
        train_identities = {row_identity(row) for row in train_rows}
        heldout_candidates = self.module.build_dataset(heldout_size * 20, heldout_seed)
        heldout_rows = select_unique_rows(
            heldout_candidates, heldout_size, excluded=train_identities
        )
        train = make_problems(train_rows, "train", f"seed={seed}", self.task_name)
        heldout = make_problems(
            heldout_rows, "heldout", f"seed={heldout_seed}", self.task_name
        )
        return train, heldout

    def heldout_record(self, problem: EpisodeProblem) -> dict[str, Any]:
        metadata = dict(problem.row["metadata"])
        return {
            "id": problem.id,
            "input": problem.input,
            "expected_running_totals": self.module.running_totals(metadata["numbers"]),
            "metadata": metadata,
            "generator_seed": problem.seed,
        }


class NumberGuessAdapter(ImportedMultiTurnAdapter):
    task_name = "structured-number-guess-grpo"
    environment_path = TASK_MODULES[task_name]
    source_description = {
        "generator": "environment.build_dataset",
        "split_scheme": (
            "the environment generator has fixed rng seed 7, so each episode uses a "
            "nonoverlapping 100-integer range block; train blocks precede held-out blocks"
        ),
    }

    def load_environment(self) -> Any:
        return self.module.load_environment(num_examples=1, max_turns=7)

    def build_splits(
        self, train_size: int, heldout_size: int, seed: int
    ) -> tuple[list[EpisodeProblem], list[EpisodeProblem]]:
        block_offset = seed * 1000
        all_problems: list[EpisodeProblem] = []
        for index in range(train_size + heldout_size):
            low = block_offset + index * 100 + 1
            high = low + 99
            row = self.module.build_dataset(
                num_examples=1, low=low, high=high, max_turns=7
            )[0]
            split = "train" if index < train_size else "heldout"
            row = dict(row)
            row["id"] = f"number-guess-{split}-{index:04d}"
            all_problems.append(
                EpisodeProblem(
                    id=row["id"],
                    row=row,
                    split=split,
                    seed=f"fixed_rng=7;range_block={index};base_seed={seed}",
                )
            )
        return all_problems[:train_size], all_problems[train_size:]

    def validate_protocol(
        self,
        problem: EpisodeProblem,
        transcript: list[dict[str, str]],
        reward: Any,
    ) -> tuple[bool, str]:
        low = int(problem.row["metadata"]["low"])
        high = int(problem.row["metadata"]["high"])
        replies = [
            str(message["content"])
            for message in transcript
            if message["role"] == "assistant"
        ]
        if not replies:
            return False, "no assistant guesses"
        if any(self.module.parse_bounded_guess(reply, low, high) is None for reply in replies):
            return False, "one or more guesses violated strict json or range"
        if not bool(reward.success):
            return False, str(reward.reason)
        return True, str(reward.reason)

    def heldout_record(self, problem: EpisodeProblem) -> dict[str, Any]:
        metadata = dict(problem.row["metadata"])
        return {
            "id": problem.id,
            "input": problem.input,
            "expected_secret": metadata["secret"],
            "metadata": metadata,
            "generator_seed": problem.seed,
        }


class MathPythonAdapter(ImportedMultiTurnAdapter):
    task_name = "math-python-grpo"
    environment_path = TASK_MODULES[task_name]
    source_description = {
        "generator": "temporary deterministic arithmetic generator",
        "split_scheme": (
            "one deterministic rng stream is generated from the base seed, then assigned "
            "by index to train followed by held-out; input digests are checked for disjointness"
        ),
        "environment_reuse": (
            "the imported environment supplies the exact system prompt, python tool execution, "
            "tool feedback, turn limit, boxed-answer parser, and reward"
        ),
    }

    def load_environment(self) -> Any:
        return self.module.load_environment(num_examples=1, max_turns=4)

    def build_splits(
        self, train_size: int, heldout_size: int, seed: int
    ) -> tuple[list[EpisodeProblem], list[EpisodeProblem]]:
        rows = generate_math_rows(train_size + heldout_size, seed, max_turns=4)
        train_rows = rows[:train_size]
        heldout_rows = rows[train_size:]
        train = make_problems(train_rows, "train", f"seed={seed}", self.task_name)
        heldout = make_problems(
            heldout_rows,
            "heldout",
            f"seed={seed};stream_offset={train_size}",
            self.task_name,
        )
        return train, heldout

    def complete_teacher(
        self,
        messages: list[dict[str, str]],
        teacher: OpenRouterTeacher,
    ) -> tuple[str, dict[str, Any]]:
        has_tool_result = any(
            message["role"] == "user" and "```output" in str(message["content"])
            for message in messages
        )
        stop = None if has_tool_result else ["\n\n\\boxed"]
        return teacher.complete(messages, stop=stop)

    def validate_protocol(
        self,
        problem: EpisodeProblem,
        transcript: list[dict[str, str]],
        reward: Any,
    ) -> tuple[bool, str]:
        assistant = [
            str(message["content"])
            for message in transcript
            if message["role"] == "assistant"
        ]
        if len(assistant) < 2:
            return False, "episode did not include a python tool turn before the final answer"
        if not any(self.module.extract_python_code(reply) is not None for reply in assistant[:-1]):
            return False, "no executable python block before the final answer"
        user_feedback = [
            str(message["content"])
            for message in transcript
            if message["role"] == "user"
        ]
        if not any("```output" in reply for reply in user_feedback):
            return False, "python tool result was not retained in the transcript"
        if self.module.extract_boxed_answer(assistant[-1]) is None:
            return False, "final assistant turn lacks a complete boxed answer"
        if not bool(reward.success):
            return False, str(reward.reason)
        return True, str(reward.reason)

    def heldout_record(self, problem: EpisodeProblem) -> dict[str, Any]:
        metadata = dict(problem.row["metadata"])
        return {
            "id": problem.id,
            "input": problem.input,
            "expected_answer": metadata["answer"],
            "metadata": metadata,
            "generator_seed": problem.seed,
        }


class SudokuAdapter(ImportedMultiTurnAdapter):
    task_name = "sudoku-grpo"
    environment_path = TASK_MODULES[task_name]
    source_description = {
        "generator": "environment.build_dataset",
        "split_scheme": (
            "train uses consecutive puzzle seeds beginning at the base seed; held-out uses "
            "consecutive seeds beginning at base seed + 1000003"
        ),
    }

    def load_environment(self) -> Any:
        return self.module.load_environment(
            num_examples=1, max_turns=30, seed=42, difficulty="easy"
        )

    def complete_teacher(
        self,
        messages: list[dict[str, str]],
        teacher: OpenRouterTeacher,
    ) -> tuple[str, dict[str, Any]]:
        response, usage = teacher.complete(messages, stop=["</move>"])
        return response.rstrip() + "</move>", usage

    def call_step(
        self,
        example: TaskExample,
        messages_before_response: list[dict[str, str]],
        assistant_response: str,
    ) -> Any:
        return self.environment.step_episode(
            example, messages_before_response, assistant_response
        )

    def build_splits(
        self, train_size: int, heldout_size: int, seed: int
    ) -> tuple[list[EpisodeProblem], list[EpisodeProblem]]:
        heldout_seed = seed + 1_000_003
        train_rows = self.module.build_dataset(
            train_size, max_turns=30, seed=seed, difficulty="easy"
        )
        heldout_rows = self.module.build_dataset(
            heldout_size, max_turns=30, seed=heldout_seed, difficulty="easy"
        )
        train = make_problems(
            train_rows, "train", f"seed_range={seed}..{seed + train_size - 1}", self.task_name
        )
        heldout = make_problems(
            heldout_rows,
            "heldout",
            f"seed_range={heldout_seed}..{heldout_seed + heldout_size - 1}",
            self.task_name,
        )
        return train, heldout

    def validate_protocol(
        self,
        problem: EpisodeProblem,
        transcript: list[dict[str, str]],
        reward: Any,
    ) -> tuple[bool, str]:
        assistant = [
            str(message["content"])
            for message in transcript
            if message["role"] == "assistant"
        ]
        for reply in assistant:
            if len(self.module.re.findall(r"<move>.*?</move>", reply, self.module.re.I | self.module.re.S)) != 1:
                return False, "assistant turn did not contain exactly one move delimiter"
            if not reply.strip().lower().endswith("</move>"):
                return False, "assistant turn did not end at the move delimiter"
            move = self.module.parse_move_string(self.module.extract_move(reply))
            if move is None:
                return False, "assistant turn had an invalid move"
        if not bool(reward.success):
            return False, str(reward.reason)
        example = self.task_example(problem)
        board = self.module._replay_board(example, tuple(transcript))
        solution = problem.row["metadata"]["solution"]
        if board.board != solution:
            return False, "solved reward did not reproduce the stored unique solution"
        return True, str(reward.reason)

    def heldout_record(self, problem: EpisodeProblem) -> dict[str, Any]:
        metadata = dict(problem.row["metadata"])
        return {
            "id": problem.id,
            "input": problem.input,
            "expected_solution": metadata["solution"],
            "metadata": metadata,
            "generator_seed": problem.seed,
        }


ADAPTERS = {
    "running-total-sft": RunningTotalAdapter,
    "structured-number-guess-grpo": NumberGuessAdapter,
    "math-python-grpo": MathPythonAdapter,
    "sudoku-grpo": SudokuAdapter,
}


def row_identity(row: dict[str, Any]) -> str:
    identity = {
        "input": str(row["input"]),
        "metadata": dict(row.get("metadata") or {}),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def select_unique_rows(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    excluded: set[str] | None = None,
) -> list[dict[str, Any]]:
    seen = set(excluded or ())
    selected: list[dict[str, Any]] = []
    for row in candidates:
        identity = row_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f"only found {len(selected)} unique rows; needed {count}")


def make_problems(
    rows: list[dict[str, Any]], split: str, seed: str, task_name: str
) -> list[EpisodeProblem]:
    problems: list[EpisodeProblem] = []
    for index, original in enumerate(rows):
        row = dict(original)
        row["metadata"] = dict(row.get("metadata") or {})
        identifier = f"{task_name}-{split}-{index:04d}"
        row["id"] = identifier
        problems.append(
            EpisodeProblem(id=identifier, row=row, split=split, seed=seed)
        )
    return problems


def generate_math_rows(count: int, seed: int, max_turns: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(rows) < count:
        family = len(rows) % 8
        if family == 0:
            left = rng.randint(10**8, 10**11)
            right = rng.randint(10**6, 10**9)
            question = f"Compute exactly: {left} multiplied by {right}."
            answer = left * right
        elif family == 1:
            base = rng.randint(10**5, 10**9)
            exponent = rng.randint(500, 5000)
            modulus = rng.randint(100_003, 999_983)
            question = f"Compute {base}^{exponent} modulo {modulus}."
            answer = pow(base, exponent, modulus)
        elif family == 2:
            left = rng.randint(10**9, 10**13)
            right = rng.randint(10**9, 10**13)
            question = f"Find gcd({left}, {right})."
            answer = math.gcd(left, right)
        elif family == 3:
            left = rng.randint(100, 5000)
            right = rng.randint(100, 5000)
            question = f"Find lcm({left}, {right})."
            answer = math.lcm(left, right)
        elif family == 4:
            n = rng.randint(40, 180)
            k = rng.randint(5, min(30, n - 5))
            question = f"Compute the binomial coefficient C({n}, {k})."
            answer = math.comb(n, k)
        elif family == 5:
            base = rng.randint(20, 200)
            exponent = rng.randint(20, 120)
            question = f"Find the sum of the decimal digits of {base}^{exponent}."
            answer = sum(int(digit) for digit in str(base**exponent))
        elif family == 6:
            n = rng.randint(25, 150)
            question = f"Find the sum of the decimal digits of {n} factorial."
            answer = sum(int(digit) for digit in str(math.factorial(n)))
        else:
            n = rng.randint(10**5, 10**8)
            power = rng.randint(2, 7)
            question = f"Find the sum of the decimal digits of {n}^{power}."
            answer = sum(int(digit) for digit in str(n**power))
        if question in seen:
            continue
        seen.add(question)
        index = len(rows)
        rows.append(
            {
                "id": f"math-python-generated-{index:04d}",
                "input": question,
                "output": f"\\boxed{{{answer}}}",
                "metadata": {
                    "answer": str(answer),
                    "max_turns": max_turns,
                    "generator_seed": seed,
                    "generator_index": index,
                },
            }
        )
    return rows


def problem_digest(problem: EpisodeProblem) -> str:
    identity = {
        "input": problem.input,
        "metadata": problem.row.get("metadata") or {},
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assert_disjoint(
    train: list[EpisodeProblem], heldout: list[EpisodeProblem]
) -> None:
    train_digests = {problem_digest(problem) for problem in train}
    heldout_digests = {problem_digest(problem) for problem in heldout}
    overlap = train_digests & heldout_digests
    if overlap:
        raise ValueError(f"train and held-out episodes overlap in {len(overlap)} rows")


def add_usage(target: Usage, source: Usage) -> None:
    target.prompt_tokens += source.prompt_tokens
    target.cached_prompt_tokens += source.cached_prompt_tokens
    target.completion_tokens += source.completion_tokens
    target.total_tokens += source.total_tokens
    target.cost_usd += source.cost_usd
    target.upstream_inference_cost_usd += source.upstream_inference_cost_usd


def generate_one(
    problem: EpisodeProblem,
    *,
    adapter: EpisodeAdapter,
    teacher: OpenRouterTeacher,
    generation_attempts: int,
) -> EpisodeResult:
    total_usage = Usage()
    reasons: list[str] = []
    attempt_records: list[dict[str, Any]] = []
    last_score = 0.0
    last_turns = 0
    for attempt_index in range(1, generation_attempts + 1):
        try:
            attempt = adapter.drive(problem, teacher)
            add_usage(total_usage, attempt.usage)
            last_score = attempt.reward_score
            last_turns = attempt.turns
            attempt_records.append(
                {
                    "attempt": attempt_index,
                    "kept": attempt.kept,
                    "reason": attempt.reason,
                    "reward_score": attempt.reward_score,
                    "turns": attempt.turns,
                    "usage": asdict(attempt.usage),
                }
            )
            if attempt.kept:
                return EpisodeResult(
                    problem=problem,
                    kept=True,
                    transcript=attempt.transcript,
                    attempts=attempt_index,
                    reasons=reasons,
                    reward_score=attempt.reward_score,
                    turns=attempt.turns,
                    usage=total_usage,
                    attempt_records=attempt_records,
                )
            reasons.append(attempt.reason)
        except RuntimeError as error:
            reason = str(error)
            reasons.append(reason)
            attempt_records.append(
                {
                    "attempt": attempt_index,
                    "kept": False,
                    "reason": reason,
                    "reward_score": 0.0,
                    "turns": 0,
                    "usage": asdict(Usage()),
                }
            )
    return EpisodeResult(
        problem=problem,
        kept=False,
        transcript=None,
        attempts=generation_attempts,
        reasons=reasons,
        reward_score=last_score,
        turns=last_turns,
        usage=total_usage,
        attempt_records=attempt_records,
    )


def transcript_characters(transcript: list[dict[str, str]]) -> int:
    return sum(len(str(message["content"])) for message in transcript)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = ADAPTERS[args.task]()
    train_problems, heldout_problems = adapter.build_splits(
        args.train_size, args.heldout_size, args.seed
    )
    assert_disjoint(train_problems, heldout_problems)

    env_values = load_env_file(args.api_env_file)
    api_key = os.environ.get(args.api_key_var) or env_values.get(args.api_key_var)
    if not api_key:
        raise ValueError(f"missing {args.api_key_var} in the environment or env file")
    teacher_model = args.teacher_model or TEACHERS[args.task]
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS[args.task]
    teacher = OpenRouterTeacher(
        endpoint=args.endpoint,
        model=teacher_model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=args.temperature,
        reasoning_enabled=False,
        http_retries=args.http_retries,
        backoff_seconds=args.backoff_seconds,
    )

    results: list[EpisodeResult] = []
    attempts_path = output_dir / "attempts.jsonl"
    with (
        attempts_path.open("w") as attempts_file,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor,
    ):
        future_to_problem = {
            executor.submit(
                generate_one,
                problem,
                adapter=adapter,
                teacher=teacher,
                generation_attempts=args.generation_attempts,
            ): problem
            for problem in train_problems
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_problem), start=1
        ):
            result = future.result()
            results.append(result)
            attempts_file.write(
                json.dumps(
                    {
                        "id": result.problem.id,
                        "kept": result.kept,
                        "attempts": result.attempts,
                        "reasons": result.reasons,
                        "reward_score": result.reward_score,
                        "turns": result.turns,
                        "usage": asdict(result.usage),
                        "attempt_records": result.attempt_records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            attempts_file.flush()
            kept_so_far = sum(item.kept for item in results)
            print(
                f"completed {completed}/{len(train_problems)}; kept={kept_so_far}",
                flush=True,
            )

    by_id = {result.problem.id: result for result in results}
    ordered = [by_id[problem.id] for problem in train_problems]
    kept = [result for result in ordered if result.kept]
    with (output_dir / "train.jsonl").open("w") as train_file:
        for result in kept:
            train_file.write(
                json.dumps(
                    {
                        "input": result.problem.input,
                        "output": {"messages": result.transcript},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    write_json(
        output_dir / "heldout.json",
        [adapter.heldout_record(problem) for problem in heldout_problems],
    )

    total_usage = Usage()
    for result in ordered:
        add_usage(total_usage, result.usage)
    turns = [result.turns for result in kept]
    lengths = [transcript_characters(result.transcript or []) for result in kept]
    manifest = {
        "task": args.task,
        "environment_module": str(adapter.environment_path.relative_to(REPO_ROOT)),
        "source": adapter.source_description,
        "teacher": {
            "endpoint": args.endpoint,
            "model": teacher_model,
            "temperature": args.temperature,
            "max_tokens": max_tokens,
            "reasoning_enabled": False,
        },
        "split": {
            "requested_train": len(train_problems),
            "kept_train": len(kept),
            "heldout": len(heldout_problems),
            "train_heldout_overlap": 0,
            "heldout_teacher_requests": 0,
            "base_seed": args.seed,
        },
        "rejection_sampling": {
            "generation_attempts": args.generation_attempts,
            "rejected": len(ordered) - len(kept),
            "verified_yield": len(kept) / len(ordered) if ordered else 0.0,
            "rejections": [
                {
                    "id": result.problem.id,
                    "reasons": result.reasons,
                    "attempts": result.attempts,
                }
                for result in ordered
                if not result.kept
            ],
        },
        "episode_statistics": {
            "mean_assistant_turns": statistics.fmean(turns) if turns else 0.0,
            "min_assistant_turns": min(turns) if turns else 0,
            "max_assistant_turns": max(turns) if turns else 0,
            "mean_transcript_characters": statistics.fmean(lengths) if lengths else 0.0,
        },
        "usage": asdict(total_usage),
        "files": {
            "train": "train.jsonl",
            "heldout": "heldout.json",
            "attempts": "attempts.jsonl",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "task": args.task,
                "kept": len(kept),
                "total": len(ordered),
                "usage": asdict(total_usage),
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-env-file", type=Path)
    parser.add_argument("--api-key-var", default="OPENROUTER_API_KEY")
    parser.add_argument("--endpoint", default=OPENROUTER_ENDPOINT)
    parser.add_argument("--teacher-model")
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--heldout-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--generation-attempts", type=int, default=2)
    parser.add_argument("--http-retries", type=int, default=5)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
