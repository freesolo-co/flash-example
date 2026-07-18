#!/usr/bin/env python3
"""Generate reward-verified single-turn distillation data."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from freesolo.datasets import TaskExample

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
GSM8K_URLS = {
    "train": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/train.jsonl"
    ),
    "test": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/test.jsonl"
    ),
}
OPENBOOKQA_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"
OPENBOOKQA_PATHS = {
    "train": "OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl",
    "test": "OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl",
}
TEACHERS = {
    "logic-boolean-grpo": "z-ai/glm-5.2",
    "math-boxed-grpo": "moonshotai/kimi-k2.6",
    "thinking-math-opd": "moonshotai/kimi-k2.6",
    "thinking-science-grpo": "moonshotai/kimi-k2.6",
}
TASK_ENVIRONMENTS = {
    task: EXAMPLES_ROOT / task / "environment.py" for task in TEACHERS
}
_GSM8K_ANSWER = re.compile(r"####\s*([^\n]+)\s*$")


@dataclass(frozen=True)
class Problem:
    id: str
    input: str
    answer: str
    source_split: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    upstream_inference_cost_usd: float = 0.0

    def add(self, payload: dict[str, Any]) -> None:
        prompt_details = dict(payload.get("prompt_tokens_details") or {})
        cost_details = dict(payload.get("cost_details") or {})
        self.prompt_tokens += int(payload.get("prompt_tokens") or 0)
        self.cached_prompt_tokens += int(prompt_details.get("cached_tokens") or 0)
        self.completion_tokens += int(payload.get("completion_tokens") or 0)
        self.total_tokens += int(payload.get("total_tokens") or 0)
        self.cost_usd += float(payload.get("cost") or 0.0)
        self.upstream_inference_cost_usd += float(
            cost_details.get("upstream_inference_cost") or 0.0
        )


@dataclass
class GenerationResult:
    problem: Problem
    kept: bool
    output: str | None
    attempts: int
    reasons: list[str]
    usage: Usage


class SingleTurnAdapter:
    def __init__(self, task: str) -> None:
        self.task = task
        self.module = load_module(TASK_ENVIRONMENTS[task])
        self.environment = self.module.load_environment()

    def output_for_answer(self, answer: str) -> str:
        if self.task == "math-boxed-grpo":
            return f"\\boxed{{{answer}}}"
        if self.task == "thinking-math-opd":
            return f"Answer: {answer}"
        if self.task == "logic-boolean-grpo":
            return f"<answer>{answer}</answer>"
        if self.task == "thinking-science-grpo":
            return f"Answer: {answer}"
        raise ValueError(f"unsupported task: {self.task}")

    def task_example(self, problem: Problem) -> TaskExample:
        record = {
            "id": problem.id,
            "input": problem.input,
            "output": self.output_for_answer(problem.answer),
            "metadata": {"answer": problem.answer, "source_split": problem.source_split},
        }
        return TaskExample(
            record=record,
            id=problem.id,
            input=problem.input,
            output=record["output"],
            metadata=record["metadata"],
        )

    def prompt_messages(self, problem: Problem) -> list[dict[str, str]]:
        example = self.task_example(problem)
        return self.environment.build_prompt_messages(example, problem.input)

    def score(self, problem: Problem, response: str) -> tuple[bool, str]:
        reward = self.environment.score_response(self.task_example(problem), response)
        return bool(reward.success and reward.score == 1.0), str(reward.reason)


class OpenRouterTeacher:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        reasoning_enabled: bool,
        http_retries: int,
        backoff_seconds: float,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_enabled = reasoning_enabled
        self.http_retries = http_retries
        self.backoff_seconds = backoff_seconds

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning": {"enabled": self.reasoning_enabled},
            "usage": {"include": True},
        }
        if stop:
            request_body["stop"] = stop
        payload = json.dumps(request_body).encode("utf-8")
        for retry in range(self.http_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "flash-example-distill/1.0",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("teacher returned an empty completion")
                return text.strip(), dict(body.get("usage") or {})
            except urllib.error.HTTPError as error:
                transient = error.code == 429 or 500 <= error.code < 600
                if not transient or retry == self.http_retries:
                    raise RuntimeError(f"teacher request failed with HTTP {error.code}") from error
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._delay(retry)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as error:
                if retry == self.http_retries:
                    raise RuntimeError(
                        "teacher request failed after transient network errors"
                    ) from error
                time.sleep(self._delay(retry))
        raise AssertionError("unreachable")

    def _delay(self, retry: int) -> float:
        jitter = random.random() * self.backoff_seconds
        return self.backoff_seconds * (2**retry) + jitter


def load_module(path: Path) -> ModuleType:
    module_name = "flash_example_distill_" + path.parent.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import environment module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "flash-example-distill/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download_jsonl(url: str) -> list[dict[str, Any]]:
    content = download_bytes(url).decode("utf-8")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def parse_gsm8k(split: str) -> list[Problem]:
    rows = download_jsonl(GSM8K_URLS[split])
    problems = []
    for index, row in enumerate(rows):
        answer_text = str(row["answer"])
        match = _GSM8K_ANSWER.search(answer_text)
        if match is None:
            raise ValueError(f"GSM8K row {split}/{index} lacks a canonical answer")
        problems.append(
            Problem(
                id=f"gsm8k-{split}-{index:05d}",
                input=str(row["question"]).strip(),
                answer=match.group(1).strip().replace(",", ""),
                source_split=split,
            )
        )
    return problems


def parse_openbookqa() -> dict[str, list[Problem]]:
    archive = zipfile.ZipFile(io.BytesIO(download_bytes(OPENBOOKQA_URL)))
    parsed: dict[str, list[Problem]] = {}
    for split, path in OPENBOOKQA_PATHS.items():
        rows = [
            json.loads(line)
            for line in archive.read(path).decode("utf-8").splitlines()
            if line.strip()
        ]
        problems = []
        for index, row in enumerate(rows):
            question = row["question"]
            choices = {choice["label"]: choice["text"] for choice in question["choices"]}
            formatted = str(question["stem"]).strip() + "\n" + "\n".join(
                f"{letter}. {choices[letter]}" for letter in "ABCD"
            )
            problems.append(
                Problem(
                    id=str(row.get("id") or f"openbookqa-{split}-{index:05d}"),
                    input=formatted,
                    answer=str(row["answerKey"]).upper(),
                    source_split=split,
                )
            )
        parsed[split] = problems
    return parsed


def generate_logic(
    module: ModuleType,
    count: int,
    seed: int,
    split: str,
    *,
    excluded_digests: set[str] | None = None,
) -> list[Problem]:
    rng = random.Random(seed)
    problems = []
    seen = set(excluded_digests or ())
    while len(problems) < count:
        depth = rng.randint(2, 5)
        expression, value = module._generate_expression(rng, depth)
        problem_input = (
            "Evaluate the following boolean expression:\n\n"
            f"{expression}\n\n"
            "Is the expression True or False?"
        )
        digest = input_digest(problem_input)
        if digest in seen:
            continue
        seen.add(digest)
        index = len(problems)
        problems.append(
            Problem(
                id=f"boolean-{split}-{index:04d}",
                input=problem_input,
                answer=str(value),
                source_split=split,
            )
        )
    return problems


def deterministic_sample(rows: list[Problem], count: int, seed: int) -> list[Problem]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a source with {len(rows)} rows")
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:count]


def input_digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def assert_disjoint(train: list[Problem], heldout: list[Problem]) -> None:
    overlap = {input_digest(row.input) for row in train} & {
        input_digest(row.input) for row in heldout
    }
    if overlap:
        raise ValueError(f"train and held-out inputs overlap in {len(overlap)} rows")


def build_splits(
    task: str,
    adapter: SingleTurnAdapter,
    train_size: int,
    heldout_size: int,
    seed: int,
) -> tuple[list[Problem], list[Problem], dict[str, Any]]:
    if task in {"math-boxed-grpo", "thinking-math-opd"}:
        train = deterministic_sample(parse_gsm8k("train"), train_size, seed)
        heldout = deterministic_sample(parse_gsm8k("test"), heldout_size, seed + 1)
        source = {
            "name": "openai/gsm8k",
            "train_url": GSM8K_URLS["train"],
            "heldout_url": GSM8K_URLS["test"],
            "selection_seed": seed,
        }
    elif task == "thinking-science-grpo":
        pools = parse_openbookqa()
        train = deterministic_sample(pools["train"], train_size, seed)
        heldout = deterministic_sample(pools["test"], heldout_size, seed + 1)
        source = {
            "name": "allenai/openbookqa",
            "archive_url": OPENBOOKQA_URL,
            "train_split": OPENBOOKQA_PATHS["train"],
            "heldout_split": OPENBOOKQA_PATHS["test"],
            "selection_seed": seed,
        }
    elif task == "logic-boolean-grpo":
        train = generate_logic(adapter.module, train_size, seed, "train")
        heldout = generate_logic(
            adapter.module,
            heldout_size,
            seed + 1,
            "heldout",
            excluded_digests={input_digest(problem.input) for problem in train},
        )
        source = {
            "name": "environment recursive boolean generator",
            "train_seed": seed,
            "heldout_seed": seed + 1,
            "depth_range": [2, 5],
        }
    else:
        raise ValueError(f"unsupported task: {task}")
    assert_disjoint(train, heldout)
    return train, heldout, source


def generate_one(
    problem: Problem,
    *,
    adapter: SingleTurnAdapter,
    teacher: OpenRouterTeacher,
    generation_attempts: int,
) -> GenerationResult:
    usage = Usage()
    reasons = []
    for attempt in range(1, generation_attempts + 1):
        try:
            response, request_usage = teacher.complete(adapter.prompt_messages(problem))
            usage.add(request_usage)
            correct, reason = adapter.score(problem, response)
            if correct:
                return GenerationResult(problem, True, response, attempt, reasons, usage)
            reasons.append(reason)
        except RuntimeError as error:
            reasons.append(str(error))
    return GenerationResult(problem, False, None, generation_attempts, reasons, usage)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_values = load_env_file(args.api_env_file)
    api_key = os.environ.get(args.api_key_var) or env_values.get(args.api_key_var)
    if not api_key:
        raise ValueError(f"missing {args.api_key_var} in the environment or env file")

    adapter = SingleTurnAdapter(args.task)
    train, heldout, source = build_splits(
        args.task, adapter, args.train_size, args.heldout_size, args.seed
    )
    teacher_model = args.teacher_model or TEACHERS[args.task]
    max_tokens = args.max_tokens or (512 if args.task == "logic-boolean-grpo" else 384)
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

    results = []
    attempts_path = output_dir / "attempts.jsonl"
    with (
        attempts_path.open("w") as attempts_file,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor,
    ):
        futures = {
            executor.submit(
                generate_one,
                problem,
                adapter=adapter,
                teacher=teacher,
                generation_attempts=args.generation_attempts,
            ): problem
            for problem in train
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            attempts_file.write(
                json.dumps(
                    {
                        "id": result.problem.id,
                        "kept": result.kept,
                        "attempts": result.attempts,
                        "reasons": result.reasons,
                        "usage": asdict(result.usage),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            attempts_file.flush()
            print(
                f"completed {completed}/{len(train)}; kept={sum(item.kept for item in results)}",
                flush=True,
            )

    by_id = {result.problem.id: result for result in results}
    ordered = [by_id[problem.id] for problem in train]
    kept = [result for result in ordered if result.kept]
    with (output_dir / "train.jsonl").open("w") as train_file:
        for result in kept:
            train_file.write(
                json.dumps(
                    {"input": result.problem.input, "output": result.output},
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_json(
        output_dir / "heldout.json",
        [
            {
                "id": problem.id,
                "input": problem.input,
                "answer": problem.answer,
                "metadata": {
                    "answer": problem.answer,
                    "source_split": problem.source_split,
                },
            }
            for problem in heldout
        ],
    )

    total_usage = Usage()
    for result in ordered:
        total_usage.prompt_tokens += result.usage.prompt_tokens
        total_usage.cached_prompt_tokens += result.usage.cached_prompt_tokens
        total_usage.completion_tokens += result.usage.completion_tokens
        total_usage.total_tokens += result.usage.total_tokens
        total_usage.cost_usd += result.usage.cost_usd
        total_usage.upstream_inference_cost_usd += result.usage.upstream_inference_cost_usd
    write_json(
        output_dir / "manifest.json",
        {
            "task": args.task,
            "environment_module": str(TASK_ENVIRONMENTS[args.task].relative_to(REPO_ROOT)),
            "source": source,
            "teacher": {
                "endpoint": args.endpoint,
                "model": teacher_model,
                "temperature": args.temperature,
                "max_tokens": max_tokens,
                "reasoning_enabled": False,
            },
            "split": {
                "requested_train": len(train),
                "kept_train": len(kept),
                "heldout": len(heldout),
                "train_heldout_overlap": 0,
                "heldout_teacher_requests": 0,
                "base_seed": args.seed,
            },
            "rejection_sampling": {
                "generation_attempts": args.generation_attempts,
                "rejected": len(ordered) - len(kept),
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
            "usage": asdict(total_usage),
            "files": {
                "train": "train.jsonl",
                "heldout": "heldout.json",
                "attempts": "attempts.jsonl",
            },
        },
    )
    print(json.dumps({"kept": len(kept), "total": len(ordered)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TEACHERS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-env-file", type=Path)
    parser.add_argument("--api-key-var", default="OPENROUTER_API_KEY")
    parser.add_argument("--endpoint", default=OPENROUTER_ENDPOINT)
    parser.add_argument("--teacher-model")
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--heldout-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--http-retries", type=int, default=5)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
