"""Self-contained multi-turn Sudoku environment for GRPO."""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any

from freesolo.datasets import TaskExample
from freesolo.environments import (
    EnvironmentEpisode,
    EnvironmentMultiTurn,
    EnvironmentStepResult,
    RewardResult,
)

_DATASET_PATH = Path(__file__).parent / "data" / "train.jsonl"

THINK_SUDOKU_SYSTEM_PROMPT = """You solve Sudoku one move at a time and reason carefully before acting.

Rules:
- Replace every empty cell, shown as ., with a digit from 1 through 9.
- Every row, column, and 3 by 3 box must contain each digit exactly once.
- Make exactly one move per assistant turn.

Reason inside <think>...</think>. Check row, column, and box constraints, prefer cells with few candidates, and backtrack when needed.

End every response with exactly one action in this form: <move>A1=5</move>
- Rows are A through I from top to bottom.
- Columns are 1 through 9 from left to right.
- Digits 1 through 9 place a value.
- Digit 0 clears a cell, for example <move>A1=0</move>.
"""


class SudokuBoard:
    """Mutable Sudoku board with move and violation tracking."""

    def __init__(self, initial_board: list[list[int]] | None = None) -> None:
        if initial_board is None:
            self.board = [[0 for _ in range(9)] for _ in range(9)]
        else:
            self.board = copy.deepcopy(initial_board)
        self.initial_board = copy.deepcopy(self.board)
        self.move_history: list[tuple[int, int, int]] = []
        self.constraint_violations = 0

    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        if not (0 <= row < 9 and 0 <= col < 9 and 1 <= num <= 9):
            return False
        if self.board[row][col] != 0:
            return False
        return not self._in_row(row, num) and not self._in_col(col, num) and not self._in_box(row, col, num)

    def make_move(self, row: int, col: int, num: int) -> bool:
        if not (0 <= row < 9 and 0 <= col < 9):
            return False
        if self.initial_board[row][col] != 0:
            return False
        if num == 0:
            if self.board[row][col] == 0:
                return False
            self.board[row][col] = 0
            self.move_history.append((row, col, 0))
            return True
        if self.is_valid_move(row, col, num):
            self.board[row][col] = num
            self.move_history.append((row, col, num))
            return True
        self.constraint_violations += 1
        return False

    def undo_last_move(self) -> bool:
        if not self.move_history:
            return False
        row, col, _ = self.move_history.pop()
        self.board[row][col] = 0
        return True

    def is_complete(self) -> bool:
        return all(0 not in row for row in self.board)

    def is_solved(self) -> bool:
        if not self.is_complete():
            return False
        expected = set(range(1, 10))
        if any(set(row) != expected for row in self.board):
            return False
        if any({self.board[row][col] for row in range(9)} != expected for col in range(9)):
            return False
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                values = {
                    self.board[row][col]
                    for row in range(box_row, box_row + 3)
                    for col in range(box_col, box_col + 3)
                }
                if values != expected:
                    return False
        return True

    def get_empty_cells(self) -> list[tuple[int, int]]:
        return [
            (row, col)
            for row in range(9)
            for col in range(9)
            if self.board[row][col] == 0
        ]

    def get_possible_values(self, row: int, col: int) -> set[int]:
        if self.board[row][col] != 0:
            return set()
        return {
            num
            for num in range(1, 10)
            if not self._in_row(row, num)
            and not self._in_col(col, num)
            and not self._in_box(row, col, num)
        }

    def _in_row(self, row: int, num: int) -> bool:
        return num in self.board[row]

    def _in_col(self, col: int, num: int) -> bool:
        return any(self.board[row][col] == num for row in range(9))

    def _in_box(self, row: int, col: int, num: int) -> bool:
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        return any(
            self.board[box_row][box_col] == num
            for box_row in range(start_row, start_row + 3)
            for box_col in range(start_col, start_col + 3)
        )

    def to_string(self) -> str:
        lines = ["  1 2 3 | 4 5 6 | 7 8 9"]
        row_labels = "ABCDEFGHI"
        for row_index, row in enumerate(self.board):
            if row_index in (3, 6):
                lines.append("  ------+-------+------")
            cells: list[str] = []
            for col_index, value in enumerate(row):
                if col_index in (3, 6):
                    cells.append("|")
                cells.append("." if value == 0 else str(value))
            lines.append(f"{row_labels[row_index]} " + " ".join(cells))
        return "\n".join(lines)

    def get_progress_stats(self) -> dict[str, int | float]:
        filled_cells = sum(cell != 0 for row in self.board for cell in row)
        initial_filled = sum(cell != 0 for row in self.initial_board for cell in row)
        return {
            "total_cells": 81,
            "filled_cells": filled_cells,
            "empty_cells": 81 - filled_cells,
            "initial_clues": initial_filled,
            "player_moves": filled_cells - initial_filled,
            "completion_percentage": filled_cells / 81 * 100,
            "moves_made": len(self.move_history),
            "constraint_violations": self.constraint_violations,
        }


def _is_valid_placement(board: list[list[int]], row: int, col: int, num: int) -> bool:
    if any(board[row][index] == num or board[index][col] == num for index in range(9)):
        return False
    start_row = row - row % 3
    start_col = col - col % 3
    return all(
        board[start_row + row_offset][start_col + col_offset] != num
        for row_offset in range(3)
        for col_offset in range(3)
    )


def _fill_board(board: list[list[int]], rng: random.Random) -> bool:
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                candidates = list(range(1, 10))
                rng.shuffle(candidates)
                for num in candidates:
                    if _is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if _fill_board(board, rng):
                            return True
                        board[row][col] = 0
                return False
    return True


def solve_sudoku(puzzle: list[list[int]]) -> list[list[int]]:
    solution = copy.deepcopy(puzzle)

    def solve() -> bool:
        for row in range(9):
            for col in range(9):
                if solution[row][col] == 0:
                    for num in range(1, 10):
                        if _is_valid_placement(solution, row, col, num):
                            solution[row][col] = num
                            if solve():
                                return True
                            solution[row][col] = 0
                    return False
        return True

    if not solve():
        raise ValueError("generated Sudoku puzzle is not solvable")
    return solution


def _count_solutions(puzzle: list[list[int]], limit: int = 2) -> int:
    board = copy.deepcopy(puzzle)
    count = 0

    def search() -> None:
        nonlocal count
        if count >= limit:
            return
        empty: tuple[int, int] | None = None
        candidates: set[int] | None = None
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    values = {
                        num
                        for num in range(1, 10)
                        if _is_valid_placement(board, row, col, num)
                    }
                    if not values:
                        return
                    if candidates is None or len(values) < len(candidates):
                        empty = (row, col)
                        candidates = values
        if empty is None or candidates is None:
            count += 1
            return
        row, col = empty
        for num in sorted(candidates):
            board[row][col] = num
            search()
            board[row][col] = 0
            if count >= limit:
                return

    search()
    return count


def _blank_range(difficulty: str) -> tuple[int, int]:
    ranges = {
        "very_easy": (6, 8),
        "easy": (8, 12),
        "medium": (13, 16),
        "hard": (17, 20),
    }
    try:
        return ranges[difficulty]
    except KeyError as error:
        choices = ", ".join(sorted(ranges))
        raise ValueError(f"difficulty must be one of: {choices}") from error


def generate_puzzle(difficulty: str = "easy", seed: int = 42) -> tuple[list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    solution = [[0 for _ in range(9)] for _ in range(9)]
    if not _fill_board(solution, rng):
        raise RuntimeError("failed to generate a complete Sudoku board")

    puzzle = copy.deepcopy(solution)
    minimum_blanks, maximum_blanks = _blank_range(difficulty)
    target_blanks = rng.randint(minimum_blanks, maximum_blanks)
    cells = [(row, col) for row in range(9) for col in range(9)]
    rng.shuffle(cells)
    removed = 0
    for row, col in cells:
        if removed >= target_blanks:
            break
        previous = puzzle[row][col]
        puzzle[row][col] = 0
        if _count_solutions(puzzle) == 1:
            removed += 1
        else:
            puzzle[row][col] = previous
    if removed != target_blanks:
        raise RuntimeError("failed to generate the requested bounded puzzle")

    solved = solve_sudoku(puzzle)
    if solved != solution or not SudokuBoard(solved).is_solved():
        raise RuntimeError("generated puzzle solution failed validation")
    return puzzle, solved


def extract_move(text: str) -> str:
    matches = list(re.finditer(r"<move>(.*?)</move>", text, re.DOTALL | re.IGNORECASE))
    if len(matches) != 1 or text[matches[0].end() :].strip():
        return ""
    return matches[0].group(1).strip()


def parse_move_string(move_text: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"([A-I])([1-9])=([0-9])", move_text.strip().upper())
    if match is None:
        return None
    row_letter, col_text, num_text = match.groups()
    return ord(row_letter) - ord("A"), int(col_text) - 1, int(num_text)


def format_move(row: int, col: int, num: int) -> str:
    return f"{chr(ord('A') + row)}{col + 1}={num}"


def calculate_move_reward(
    board: SudokuBoard,
    move_valid: bool,
    move_info: dict[str, Any] | None = None,
) -> float:
    details = move_info or {}
    if not move_valid:
        return -0.2
    if details.get("is_backtrack", False):
        return -0.1
    reward = 0.1
    if board.is_solved():
        return reward + 2.0
    reward += float(board.get_progress_stats()["completion_percentage"]) / 1000
    if "possibilities_before" in details and "possibilities_after" in details:
        reduction = details["possibilities_before"] - details["possibilities_after"]
        if reduction > 0:
            reward += min(0.05, reduction * 0.01)
    if details.get("completed_constraint", False):
        reward += 0.5
    return reward


def calculate_final_reward(
    board: SudokuBoard,
    max_turns: int,
    turns_used: int,
    *,
    solved: bool | None = None,
) -> float:
    solved = board.is_solved() if solved is None else solved
    if solved:
        reward = 5.0
        if max_turns > 0:
            reward += 1.0 - turns_used / max_turns
    elif board.is_complete():
        reward = 1.0
    else:
        completion = float(board.get_progress_stats()["completion_percentage"]) / 100
        reward = completion * 2.0
    reward -= board.constraint_violations * 0.1
    return max(0.0, reward)


def get_move_feedback(
    board: SudokuBoard,
    move_valid: bool,
    row: int | None = None,
    col: int | None = None,
    num: int | None = None,
    *,
    solved: bool | None = None,
) -> str:
    solved = board.is_solved() if solved is None else solved
    if move_valid:
        if num == 0:
            return f"Backtracked. Cell cleared. {board.get_progress_stats()['empty_cells']} cells remaining."
        if solved:
            return "Congratulations. You solved the puzzle."
        if board.is_complete():
            return "The board is complete but incorrect. Review the entries."
        return f"Valid move. {board.get_progress_stats()['empty_cells']} cells remaining."

    if row is not None and col is not None and num is not None:
        row_letter = chr(ord("A") + row)
        col_number = col + 1
        if num == 0 and board.board[row][col] == 0:
            return f"Cell {row_letter}{col_number} is already empty."
        if num != 0 and board.board[row][col] != 0:
            return f"Cell {row_letter}{col_number} is already filled."
        if num != 0 and board._in_row(row, num):
            return f"Number {num} already exists in row {row_letter}."
        if num != 0 and board._in_col(col, num):
            return f"Number {num} already exists in column {col_number}."
        if num != 0 and board._in_box(row, col, num):
            return f"Number {num} already exists in box {row // 3 + 1}{col // 3 + 1}."
    return "Invalid move. Check the Sudoku constraints."


def _initial_prompt(board: SudokuBoard) -> str:
    return f"""Sudoku puzzle:

{board.to_string()}

Fill the empty cells so every row, column, and 3 by 3 box contains 1 through 9 exactly once.
Make one move per turn and end with the format <move>A1=5</move>. Use 0 to clear a cell."""


def build_dataset(
    num_examples: int = 24,
    max_turns: int = 30,
    seed: int = 42,
    difficulty: str = "easy",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(num_examples):
        puzzle, solution = generate_puzzle(difficulty=difficulty, seed=seed + index)
        board = SudokuBoard(puzzle)
        stats = board.get_progress_stats()
        rows.append(
            {
                "id": f"sudoku-{index:03d}",
                "input": _initial_prompt(board),
                "output": "Solve the unique puzzle one move at a time.",
                "metadata": {
                    "puzzle": puzzle,
                    "solution": solution,
                    "difficulty": difficulty,
                    "initial_clues": stats["initial_clues"],
                    "total_moves_needed": stats["empty_cells"],
                    "max_turns": max_turns,
                },
            }
        )
    return rows


def load_distilled_dataset(path: str | Path = _DATASET_PATH) -> list[dict[str, Any]]:
    generated = {
        str(row["input"]): row
        for row in build_dataset(
            num_examples=100,
            max_turns=30,
            seed=20260717,
            difficulty="easy",
        )
    }
    rows = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row.get("metadata") or {})
            if not {"puzzle", "solution", "max_turns"} <= metadata.keys():
                source = generated.get(str(row["input"]))
                if source is None:
                    raise ValueError(
                        f"distilled row {index} does not include generator metadata"
                    )
                metadata = dict(source["metadata"])
            rows.append(
                {
                    "id": f"sudoku-distilled-{index:04d}",
                    "input": row["input"],
                    "output": row["output"],
                    "metadata": metadata,
                }
            )
    return rows


def _metadata(example: TaskExample) -> dict[str, Any]:
    values = dict(example.metadata or {})
    if "puzzle" not in values or "solution" not in values or "max_turns" not in values:
        raise ValueError("Sudoku example metadata is incomplete")
    return values


def _replay_board(example: TaskExample, messages: list[dict[str, str]] | tuple[dict[str, str], ...]) -> SudokuBoard:
    board = SudokuBoard(_metadata(example)["puzzle"])
    for message in messages:
        if message["role"] != "assistant":
            continue
        move = parse_move_string(extract_move(str(message["content"])))
        if move is not None:
            board.make_move(*move)
    return board


def _episode_reply(board: SudokuBoard, feedback: str, turn: int, max_turns: int) -> str:
    stats = board.get_progress_stats()
    return f"""{feedback}

Current board:
{board.to_string()}

Progress: {stats['filled_cells']}/81 cells filled ({stats['completion_percentage']:.1f}%)
Turn: {turn}/{max_turns}

What is your next move?"""


class SudokuEnvironment(EnvironmentMultiTurn):
    def __init__(self, dataset_path: str | Path = _DATASET_PATH) -> None:
        self.dataset = load_distilled_dataset(dataset_path)

    def start_episode(
        self,
        example: TaskExample,
        prompt_text: str,
    ) -> list[dict[str, str]]:
        _ = prompt_text
        return [
            {"role": "system", "content": THINK_SUDOKU_SYSTEM_PROMPT},
            {"role": "user", "content": str(example.input)},
        ]

    def max_episode_turns(self, example: TaskExample) -> int:
        return int(_metadata(example)["max_turns"])

    def step_episode(
        self,
        example: TaskExample,
        messages: list[dict[str, str]],
        assistant_response: str,
    ) -> EnvironmentStepResult:
        board = _replay_board(example, messages)
        turn = sum(message["role"] == "assistant" for message in messages) + 1
        move = parse_move_string(extract_move(assistant_response))
        if move is None:
            feedback = "Invalid move format. End with exactly one move such as <move>A1=5</move>."
            return EnvironmentStepResult(
                done=False,
                messages=(
                    {
                        "role": "user",
                        "content": _episode_reply(
                            board,
                            feedback,
                            turn,
                            self.max_episode_turns(example),
                        ),
                    },
                ),
            )

        row, col, num = move
        move_valid = board.make_move(row, col, num)
        solved = board.board == _metadata(example)["solution"]
        feedback = get_move_feedback(
            board, move_valid, row, col, num, solved=solved
        )
        return EnvironmentStepResult(
            done=solved,
            messages=(
                {
                    "role": "user",
                    "content": _episode_reply(
                        board,
                        feedback,
                        turn,
                        self.max_episode_turns(example),
                    ),
                },
            ),
        )

    def score_episode(
        self,
        example: TaskExample,
        episode: EnvironmentEpisode,
    ) -> RewardResult:
        board = _replay_board(example, episode.messages)
        turns_used = sum(message["role"] == "assistant" for message in episode.messages)
        solved = board.board == _metadata(example)["solution"]
        score = calculate_final_reward(
            board,
            self.max_episode_turns(example),
            turns_used,
            solved=solved,
        )
        stats = board.get_progress_stats()
        reason = (
            f"solved in {turns_used} turns"
            if solved
            else f"not solved; {stats['filled_cells']}/81 filled and {board.constraint_violations} violations"
        )
        return RewardResult(score=score, success=solved, reason=reason)

    def sft_completion(self, example: TaskExample) -> list[dict[str, str]]:
        output = example.output
        if not isinstance(output, dict) or not isinstance(output.get("messages"), list):
            raise ValueError("Sudoku SFT rows require output.messages")
        return [dict(message) for message in output["messages"]]


def load_environment(**kwargs: object) -> SudokuEnvironment:
    dataset_path = kwargs.get("dataset_path", _DATASET_PATH)
    environment = SudokuEnvironment(dataset_path=str(dataset_path))
    if "num_examples" in kwargs:
        environment.dataset = build_dataset(
            num_examples=int(kwargs["num_examples"]),
            max_turns=int(kwargs.get("max_turns", 30)),
            seed=int(kwargs.get("seed", 42)),
            difficulty=str(kwargs.get("difficulty", "easy")),
        )
    return environment
