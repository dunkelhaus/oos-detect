# import logging.config
# from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
# logging.config.dictConfig(LOGGING_CONFIG)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

# import wandb
import pandas as pd
from train.run import run_training
from train.metrics import get_metrics
from train.predict import get_predictions
from datasets.readers.oos_eval import OOSEvalReader
from datasets.readers.oos_eval import read_full_test_data
from datasets.readers.oos_eval import read_full_train_data
from models.bert_linear.builders import bert_linear_builders


# Logger setup.
# log = logging.getLogger(__name__)
# log.debug("Logging is configured.")

# wandb.init(entity="dunkelhaus", project="oos-detect",
# sync_tensorboard=True, reinit=True)

# We've copied the training loop from an earlier example, with updated
# model code, above in the Setup section. We run the training loop to get
# a trained model.

def train_test_pred_oos_full_bert_linear(run_test=False):
    dataset_reader = OOSEvalReader()
    train_data = read_full_train_data(reader=dataset_reader)
    test_data = read_full_test_data(reader=dataset_reader)

    # pp = pprint.PrettyPrinter(indent=4)
    model = run_training(
        data=train_data,
        model_builder=bert_linear_builders
    )

    if run_test:
        model = run_testing(test_data, model)

    actuals, predictions, labels = get_predictions(
        model,
        dataset_reader,
        list(test_data)
    )

    return actuals, predictions, labels

actuals, predictions, labels = train_test_pred_oos_full_bert_linear()
df = get_metrics(actuals, predictions, labels)

with pd.option_context('display.max_rows', None):
    print("\n\n=====   Multiclass Classification Report   =====")
    print(df)
