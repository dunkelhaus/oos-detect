# import logging.config
# from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
# logging.config.dictConfig(LOGGING_CONFIG)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

# import wandb
import pprint
import pandas as pd
from train.metrics import get_metrics
from train.loop import run_training_loop
from train.predict import get_test_predictions

# Logger setup.
# log = logging.getLogger(__name__)
# log.debug("Logging is configured.")

# wandb.init(entity="dunkelhaus", project="oos-detect",
# sync_tensorboard=True, reinit=True)

# We've copied the training loop from an earlier example, with updated
# model code, above in the Setup section. We run the training loop to get
# a trained model.


# pp = pprint.PrettyPrinter(indent=4)
model, dataset_reader = run_training_loop(run_test=False)

actuals, predictions, labels = get_test_predictions(
    model,
    dataset_reader
)


print("\n\n=====   Multiclass Classification Report   =====")
df = get_metrics(actuals, predictions, labels)

with pd.option_context('display.max_rows', None):
    print(df)
