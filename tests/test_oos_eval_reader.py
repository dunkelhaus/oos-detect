import pytest
import logging.config
from pathlib import Path
from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("transformers").setLevel(logging.ERROR)


@pytest.fixture
def dataset_reader():
    from datasets.readers.oos_eval import OOSEvalReader

    return OOSEvalReader()


def test_read_works(dataset_reader):
    this_src_dir = Path(__file__).parent.parent.absolute()
    dataset = dataset_reader.read(
        this_src_dir/"datasets/oos-eval/data/data_full_train.json"
    )
    print('type of dataset: ', type(dataset))
    print('type of its first element: ', type(dataset[0]))
    print(f"first element: {dataset[0].fields!r}")
    print('size of dataset: ', len(dataset))

    assert dataset is not None
