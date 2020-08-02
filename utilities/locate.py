import logging
from pathlib import Path

# Logger setup.
log = logging.getLogger(__name__)


def locate_oos_data() -> Path:
    """
    Locate OOS data, which muse be in oos-detect/datasets/oos-eval/data/.
    If not found, returns None, printing an error.

    :return Path, or None: The path to the directory, or None if not valid.
    """
    log.debug("Locating CLINC data directory.")
    path = (Path(__file__).parent.parent
        .absolute()/"datasets/oos-eval/data")

    if path.is_dir():
        return path
    else:
        log.debug("Failed! Path not resolved, or is not a file.")

    return


def locate_results_dir() -> Path:
    """
    Locate OOS data, which muse be in oos-detect/datasets/oos-eval/data/.
    If not found, returns None, printing an error.

    :return Path, or None: The path to the directory, or None if not valid.
    """
    log.debug("Locating results directory.")

    path = (Path(__file__).parent.parent
        .absolute()/"train/results")

    print(path)
    print(path.exists())

    if path.exists():
        return path
    else:
        log.debug("Failed! Please create the directory "
                        "oos-detect/train/results.")

    return
