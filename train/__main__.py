# import logging.config
# from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
# logging.config.dictConfig(LOGGING_CONFIG)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

# import wandb
import pprint
import numpy as np
from dataclasses import dataclass
from train.loop import run_training_loop
from utilities.locate import locate_oos_data
from allennlp.predictors import TextClassifierPredictor
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# Logger setup.
# log = logging.getLogger(__name__)
# log.debug("Logging is configured.")

# wandb.init(entity="dunkelhaus", project="oos-detect",
# sync_tensorboard=True, reinit=True)

# We've copied the training loop from an earlier example, with updated
# model code, above in the Setup section. We run the training loop to get
# a trained model.


@dataclass
class PerClassStats:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    total: int = 0
    accuracy: float = 0.
    precision: float = 0.
    recall: float = 0.
    f1: float = 0.


# pp = pprint.PrettyPrinter(indent=4)
model, dataset_reader = run_training_loop(run_test=False)

predictor = TextClassifierPredictor(
    model=model,
    dataset_reader=dataset_reader
)

test_data = list(
    dataset_reader.read(
        locate_oos_data()/"data_full_test.json"
    )
)

print(f"Example test instance: {test_data[0]}.")

preds = predictor.predict_batch_instance(test_data)

labelmap = predictor._model.vocab.get_index_to_token_vocabulary('labels')

print(f"Label map: {labelmap}")

print(f"Instance predictions made: {len(preds)}.")
print(f"First prediction: {preds[0]}")

predictions = [labelmap[np.argmax(l['probs'])] for l in preds]
actuals = [str(i['label'].label) for i in test_data]

labs = list(labelmap.values())
print(labs)

results = precision_recall_fscore_support(
    actuals,
    predictions,
    average=None,
    labels=labs
)



# Per-label accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)

for i in range(len(labs)):
    print("=====   Multiclass Classification Report   =====")




# print("sklearn results are below: ")
# print(results)
# pprint.pprint(per_class_matches)
