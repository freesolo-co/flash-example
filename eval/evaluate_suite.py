"""Evaluate shipped adapters on each example's frozen held-out split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEpisode
from openai import BadRequestError, OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DEFAULT_BASE_URL = "https://clado-ai--freesolo-lora-serving.modal.run/v1"
DEFAULT_CONFIG_PATH = Path.home() / ".flash" / "config.json"
UNSAFE_LOCAL_CODE_WARNING = (
    "WARNING: math-python evaluation executes model-generated Python directly on this host "
    "with no sandbox. Use only a disposable machine or container."
)


@dataclass(frozen=True)
class Profile:
    name: str
    adapter_model: str
    mode: str
    max_tokens: int
    component_name: str
    step_before_append: bool = False
    stop_sequences: tuple[str, ...] = ()


PROFILES = {
    profile.name: profile
    for profile in (
        Profile(
            "running-total-sft",
            "flash-1784320041-9c4a32b8",
            "multi",
            512,
            "all_turns_exact_rate",
        ),
        Profile(
            "logic-boolean-sft-grpo",
            "flash-1784316952-a904a84d",
            "single",
            1024,
            "exact_answer_rate",
            stop_sequences=("</answer>",),
        ),
        Profile(
            "structured-number-guess-sft-grpo",
            "flash-1784321381-dbe68392",
            "multi",
            128,
            "solved_rate",
        ),
        Profile(
            "thinking-science-opd",
            "flash-1784321193-dab97aef",
            "single",
            2048,
            "exact_answer_rate",
        ),
        Profile(
            "math-boxed-sft",
            "flash-1784263689-f98515ce",
            "single",
            2048,
            "boxed_answer_rate",
        ),
        Profile(
            "math-python-sft",
            "flash-1784322317-e152ffdd",
            "multi",
            2048,
            "exact_answer_rate",
        ),
        Profile(
            "thinking-math-sft-opd",
            "flash-1784326094-ab33c65b",
            "single",
            2048,
            "exact_answer_rate",
        ),
        Profile(
            "sudoku-sft-grpo",
            "flash-1784327754-e1d3a602",
            "multi",
            1536,
            "solved_rate",
            True,
            ("</move>",),
        ),
    )
}


@dataclass(frozen=True)
class LoadedExample:
    profile: Profile
    environment_module: ModuleType
    environment: Any
    rows: list[dict[str, Any]]
    response_schema: dict[str, Any] | None
    response_schema_factory: Callable[[dict[str, Any]], dict[str, Any]] | None


@dataclass(frozen=True)
class CompletionResult:
    content: str
    finish_reason: str | None


@dataclass(frozen=True)
class CaseResult:
    id: str
    input: str
    score: float
    success: bool
    turns: int
    truncated_turns: int
    finish_reasons: list[str | None]
    reason: str | None
    error: str | None
    response: str


@dataclass(frozen=True)
class AggregateResult:
    example: str
    model: str
    n: int
    mean_reward: float
    component_name: str
    component_rate: float
    max_tokens: int
    temperature: float
    cases: list[CaseResult]


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_modules(profile: Profile) -> tuple[ModuleType, ModuleType]:
    example_dir = EXAMPLES_ROOT / profile.name
    suffix = profile.name.replace("-", "_")
    environment_module = load_module(
        example_dir / "environment.py", f"flash_eval_{suffix}_environment"
    )
    previous_environment = sys.modules.get("environment")
    sys.modules["environment"] = environment_module
    try:
        call_module = load_module(example_dir / "call.py", f"flash_eval_{suffix}_call")
    finally:
        if previous_environment is None:
            sys.modules.pop("environment", None)
        else:
            sys.modules["environment"] = previous_environment
    return environment_module, call_module


def expected_output(name: str, row: dict[str, Any]) -> Any:
    metadata = dict(row.get("metadata") or {})
    answer = row.get("answer")
    if name == "running-total-sft":
        return {"expected_running_totals": row.get("expected_running_totals")}
    if name == "structured-number-guess-sft-grpo":
        return str(metadata["secret"])
    if name == "logic-boolean-sft-grpo":
        return f"<answer>{metadata['answer']}</answer>"
    if name == "thinking-science-opd":
        return f"Answer: {metadata['answer']}"
    if name in {"math-boxed-sft", "math-python-sft"}:
        return f"\\boxed{{{metadata.get('answer', answer)}}}"
    if name == "thinking-math-sft-opd":
        return f"Answer: {metadata.get('answer', answer)}"
    if name == "sudoku-sft-grpo":
        return "Solve the unique puzzle one move at a time."
    raise ValueError(f"unsupported example: {name}")


def load_heldout(profile: Profile) -> list[dict[str, Any]]:
    path = EXAMPLES_ROOT / profile.name / "data" / "heldout.json"
    raw_rows = json.loads(path.read_text())
    rows = []
    for row in raw_rows:
        metadata = dict(row.get("metadata") or {})
        rows.append(
            {
                "id": row["id"],
                "input": row["input"],
                "output": expected_output(profile.name, row),
                "metadata": metadata,
            }
        )
    return rows


def load_example(profile: Profile) -> LoadedExample:
    environment_module, call_module = load_modules(profile)
    rows = load_heldout(profile)
    schema = getattr(call_module, "SCHEMA", None)
    if schema is not None and not isinstance(schema, dict):
        raise TypeError("call.py SCHEMA must be a dictionary")
    schema_factory = getattr(call_module, "schema_for_metadata", None)
    if schema_factory is not None and not callable(schema_factory):
        raise TypeError("call.py schema_for_metadata must be callable")
    return LoadedExample(
        profile=profile,
        environment_module=environment_module,
        environment=environment_module.load_environment(),
        rows=rows,
        response_schema=schema,
        response_schema_factory=schema_factory,
    )


def task_example(row: dict[str, Any]) -> TaskExample:
    return TaskExample(
        record=row,
        id=str(row.get("id")) if row.get("id") is not None else None,
        input=str(row["input"]),
        output=row.get("output"),
        metadata=dict(row.get("metadata") or {}),
    )


def resolve_api_key(config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    environment_key = os.environ.get("FREESOLO_API_KEY")
    if environment_key:
        return environment_key
    try:
        config = json.loads(config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            "no usable API key: FREESOLO_API_KEY is unset and Flash CLI config could not be read"
        ) from error
    config_key = config.get("api_key")
    if not isinstance(config_key, str) or not config_key:
        raise RuntimeError(
            "no usable API key: FREESOLO_API_KEY is unset and Flash CLI config has no api_key"
        )
    return config_key


def response_schema_for_row(
    loaded: LoadedExample, row: dict[str, Any]
) -> dict[str, Any] | None:
    if loaded.response_schema_factory is not None:
        return loaded.response_schema_factory(dict(row.get("metadata") or {}))
    return loaded.response_schema


def request_completion(
    client: OpenAI,
    loaded: LoadedExample,
    model: str,
    messages: list[dict[str, str]],
    row: dict[str, Any],
) -> CompletionResult:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": loaded.profile.max_tokens,
    }
    if loaded.profile.stop_sequences:
        request["stop"] = list(loaded.profile.stop_sequences)
    response_schema = response_schema_for_row(loaded, row)
    if response_schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_response",
                "strict": True,
                "schema": response_schema,
            },
        }
    completion = client.chat.completions.create(**request)
    choice = completion.choices[0]
    content = choice.message.content or ""
    if loaded.profile.stop_sequences:
        stop = loaded.profile.stop_sequences[0]
        if not content.rstrip().endswith(stop):
            content = content.rstrip() + stop
    return CompletionResult(
        content=content,
        finish_reason=getattr(choice, "finish_reason", None),
    )


def evaluate_single_case(
    client: OpenAI, loaded: LoadedExample, model: str, row: dict[str, Any]
) -> CaseResult:
    example = task_example(row)
    messages = loaded.environment.build_prompt_messages(example, str(example.input))
    completion = request_completion(client, loaded, model, messages, row)
    reward = loaded.environment.score_response(example, completion.content)
    return CaseResult(
        id=str(example.id or ""),
        input=str(example.input),
        score=float(reward.score),
        success=bool(reward.success),
        turns=1,
        truncated_turns=1 if completion.finish_reason == "length" else 0,
        finish_reasons=[completion.finish_reason],
        reason=reward.reason,
        error=None,
        response=completion.content,
    )


def evaluate_multi_case(
    client: OpenAI, loaded: LoadedExample, model: str, row: dict[str, Any]
) -> CaseResult:
    example = task_example(row)
    messages = list(loaded.environment.start_episode(example, ""))
    last_response = ""
    finish_reasons: list[str | None] = []
    episode_error = None
    turns = 0
    for _ in range(loaded.environment.max_episode_turns(example)):
        try:
            completion = request_completion(client, loaded, model, messages, row)
        except BadRequestError as error:
            if "maximum context length" not in str(error):
                raise
            episode_error = str(error)
            break
        last_response = completion.content
        finish_reasons.append(completion.finish_reason)
        turns += 1
        if loaded.profile.step_before_append:
            step = loaded.environment.step_episode(example, messages, last_response)
            messages.append({"role": "assistant", "content": last_response})
        else:
            messages.append({"role": "assistant", "content": last_response})
            step = loaded.environment.step_episode(example, messages, last_response)
        messages.extend(step.messages)
        if step.done:
            break
    episode = EnvironmentEpisode(messages=tuple(messages), response_text=last_response)
    reward = loaded.environment.score_episode(example, episode)
    return CaseResult(
        id=str(example.id or ""),
        input=str(example.input),
        score=float(reward.score),
        success=bool(reward.success),
        turns=turns,
        truncated_turns=sum(reason == "length" for reason in finish_reasons),
        finish_reasons=finish_reasons,
        reason=reward.reason,
        error=episode_error,
        response=last_response,
    )


def evaluate_model(
    loaded: LoadedExample, model: str, workers: int, api_key: str, base_url: str
) -> AggregateResult:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=2)
    evaluator: Callable[[OpenAI, LoadedExample, str, dict[str, Any]], CaseResult]
    evaluator = evaluate_single_case if loaded.profile.mode == "single" else evaluate_multi_case

    def run(row: dict[str, Any]) -> CaseResult:
        return evaluator(client, loaded, model, row)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run, loaded.rows))
    count = len(results)
    success_rate = sum(result.success for result in results) / count
    return AggregateResult(
        example=loaded.profile.name,
        model=model,
        n=count,
        mean_reward=sum(result.score for result in results) / count,
        component_name=loaded.profile.component_name,
        component_rate=success_rate,
        max_tokens=loaded.profile.max_tokens,
        temperature=0.0,
        cases=results,
    )


def require_local_execution_opt_in(
    example: str, *, dry_run: bool, allowed: bool
) -> None:
    if example != "math-python-sft" or dry_run:
        return
    if not allowed:
        raise RuntimeError(
            "math-python evaluation is disabled because it executes model-generated Python "
            "on the host with no sandbox; rerun with --allow-unsafe-local-code-execution "
            "only inside a disposable machine or container"
        )
    print(UNSAFE_LOCAL_CODE_WARNING, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=sorted(PROFILES), required=True)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unsafe-local-code-execution",
        action="store_true",
        help=(
            "allow math-python evaluation to run model-generated Python on this host "
            "without a sandbox"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    require_local_execution_opt_in(
        args.example,
        dry_run=args.dry_run,
        allowed=args.allow_unsafe_local_code_execution,
    )
    profile = PROFILES[args.example]
    model = args.model or profile.adapter_model
    loaded = load_example(profile)
    if args.dry_run:
        payload: dict[str, Any] = {
            "example": args.example,
            "model": model,
            "n": len(loaded.rows),
            "case_ids": [str(row.get("id", "")) for row in loaded.rows],
            "max_tokens": profile.max_tokens,
            "temperature": 0.0,
        }
    else:
        result = evaluate_model(
            loaded=loaded,
            model=model,
            workers=args.workers,
            api_key=resolve_api_key(),
            base_url=args.base_url,
        )
        payload = asdict(result)
        payload["cases"] = [asdict(case) for case in result.cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "cases"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
