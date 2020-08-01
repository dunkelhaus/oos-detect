from pathlib import Path


def locate_oos_data() -> Path:
    """
    """
    path = (Path(__file__).parent.parent
        .absolute()/"datasets/oos-eval/data/data_full_train.json")

    if path.is_file():
        return path
    else:
        print("Failed! Path not resolved, or is not a file.")

    return
