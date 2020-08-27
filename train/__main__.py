#import logging.config
#from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
#logging.config.dictConfig(LOGGING_CONFIG)
#logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

#import wandb
from train.loop import run_training_loop
from allennlp.training.util import evaluate
from utilities.locate import locate_oos_data
from allennlp.predictors import TextClassifierPredictor
from allennlp.data import DataLoader, PyTorchDataLoader


# Logger setup.
#log = logging.getLogger(__name__)
#log.debug("Logging is configured.")


# wandb.init(entity="dunkelhaus", project="oos-detect", sync_tensorboard=True, reinit=True)

# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop(run_test=False)

predictor = TextClassifierPredictor(
    model=model,
    dataset_reader=dataset_reader
)

preds = predictor.predict_batch_instance(list(dataset_reader.read(locate_oos_data()/"data_full_test.json")))

print(preds)
