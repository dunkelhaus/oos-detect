# import logging.config
# from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
# logging.config.dictConfig(LOGGING_CONFIG)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

# import wandb
import pandas as pd
from oos_detect.train.run import run_testing
from oos_detect.train.run import run_training
from oos_detect.train.metrics import get_metrics
from oos_detect.train.predict import get_predictions
from oos_detect.datasets.readers.oos_eval import OOSEvalReader
from oos_detect.datasets.readers.oos_eval import oos_data_paths
from oos_detect.models.bert_linear.builders import bert_linear_builders
# from pseudo_ood_generation.components.ae.builders import pog_ae_builders


# Logger setup.
# log = logging.getLogger(__name__)
# log.debug("Logging is configured.")

# wandb.init(entity="dunkelhaus", project="oos-detect",
# sync_tensorboard=True, reinit=True)

# We've copied the training loop from an earlier example, with updated
# model code, above in the Setup section. We run the training loop to get
# a trained model.

def train_test_pred(
        builders,
        run_test=False,
        get_preds=False,
        set="full",
        model_name="dunkrun",
):
    hyperparams = {
        "num_epochs": 3,
        "batch_size": 64,
        "lr": 0.0001,
        "no_cuda": False,
        "log_interval": 10
    }
    data_reader = OOSEvalReader()
    train_paths, test_path = oos_data_paths(set=set)

    # pp = pprint.PrettyPrinter(indent=4)
    model = run_training(
        data_reader=data_reader,
        data_paths=train_paths,
        model_builder=builders,
        run_name=(model_name + "_" + set),
        hyperparams=hyperparams
    )

    if run_test:
        model = run_testing(test_path, model)

    # TODO: check test path param.
    if get_preds:
        actuals, predictions, labels = get_predictions(
            model,
            data_reader,
            list(test_path)
        )

        return actuals, predictions, labels

    return


train_test_pred(
    set="small",
    model_name="bert_linear",
    builders=bert_linear_builders
)


"""train_test_pred(
    set="small",
    model_name="pog_ae",
    builders=pog_ae_builders
)"""
# df = get_metrics(actuals, predictions, labels)

# with pd.option_context('display.max_rows', None):
#    print("\n\n=====   Multiclass Classification Report   =====")
#    print(df)
