from environment import build_dataset, parse_guess


def main() -> None:
    assert parse_guess('{"guess":42}') == 42
    assert parse_guess('{"guess":true}') is None
    assert parse_guess('guess 42') is None
    assert parse_guess('{"guess":42,"extra":1}') is None
    assert build_dataset(4) == build_dataset(4)
    print("structured number guess smoke passed")


if __name__ == "__main__":
    main()
