from pathlib import Path


def locate_oos_data() -> Path:
    """
    Locate OOS data, which muse be in oos-detect/datasets/oos-eval/data/.
    If not found, returns None, printing an error.

    :return Path, or None: The path to the directory, or None if not valid.
    """
    path = (Path(__file__).parent.parent
        .absolute()/"datasets/oos-eval/data")

    if path.is_file():
        return path
    else:
        print("Failed! Path not resolved, or is not a file.")

    return
