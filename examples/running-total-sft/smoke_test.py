from environment import build_dataset, gold_completion, load_environment, running_totals


def main() -> None:
    assert running_totals([3, 5, 2]) == [3, 8, 10]
    assert gold_completion([3, 5])[-1] == {"role": "assistant", "content": "8"}
    assert build_dataset(4, 9) == build_dataset(4, 9)
    assert build_dataset(4, 9) != build_dataset(4, 10)
    assert len(load_environment().dataset) == 100
    print("running total smoke passed with 100 distilled rows")


if __name__ == "__main__":
    main()
