from environment import build_dataset, extract_answer, load_environment


def main() -> None:
    rows = build_dataset()
    assert len(rows) == 24
    assert rows == build_dataset()
    assert extract_answer("work\nAnswer: 12") == 12
    assert extract_answer("12") is None
    assert len(load_environment().dataset) == 143
    print("thinking math smoke passed with 143 distilled rows")


if __name__ == "__main__":
    main()
