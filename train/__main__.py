import logging.config
from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

import wandb
from allennlp.data import DataLoader
from train.loop import run_training_loop
from allennlp.training.util import evaluate
from utilities.locate import locate_oos_data


# Logger setup.
log = logging.getLogger(__name__)
log.debug("Logging is configured.")


# wandb.init(entity="dunkelhaus", project="oos-detect", sync_tensorboard=True, reinit=True)

# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop()

# Now we can evaluate the model on a new dataset.
"""test_data = dataset_reader.read(locate_oos_data()/"data_full_test.json")
test_data.index_with(model.vocab)
data_loader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False
)

results = evaluate(model, data_loader)
print(results)"""
