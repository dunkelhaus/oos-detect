"""Deprecated entrypoint. Importing it does not load the legacy experiment."""


def main():
    raise SystemExit(
        "The AllenNLP/BERT training command is unsupported. "
        "Use `python -m oos_detect.reproduce --help`. "
        "The old script is preserved in oos_detect.train.legacy_bert."
    )


if __name__ == "__main__":
    main()
