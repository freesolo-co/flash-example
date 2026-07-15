from environment import build_dataset, encode_label, parse_label


def main() -> None:
    rows = build_dataset()
    assert len(rows) == 24
    assert rows == build_dataset()
    assert parse_label(rows[0]["output"]) == rows[0]["metadata"]["expected"]
    assert parse_label('{"category":"billing"}') is None
    assert encode_label(rows[0]["metadata"]["expected"]) == rows[0]["output"]
    print("json extraction smoke passed")


if __name__ == "__main__":
    main()
