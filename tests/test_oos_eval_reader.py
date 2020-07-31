import pytest
import logging.config
from configs.log import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
logging.config.dictConfig(LOGGING_CONFIG)


@pytest.fixture
def dataset_reader():
    from datasets.readers.oos_eval import OOSEvalReader

    return OOSEvalReader()


def test_read_works(dataset_reader):
    dataset = dataset_reader.read(set_type="data_full", set_portion="val")
    print('type of dataset: ', type(dataset))
    print('type of its first element: ', type(dataset[0]))
    print('size of dataset: ', len(dataset))

    assert dataset is not None
