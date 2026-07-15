from environment import build_dataset, extract_answer


def main() -> None:
    rows = build_dataset()
    assert len(rows) == 24
    assert rows == build_dataset()
    assert extract_answer("reasoning\nAnswer: C") == "C"
    assert extract_answer("C") is None
    print("thinking science smoke passed")


if __name__ == "__main__":
    main()
