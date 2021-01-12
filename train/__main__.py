# import logging.config
# from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
# logging.config.dictConfig(LOGGING_CONFIG)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

# import wandb
import numpy as np
from train.loop import run_training_loop
from utilities.locate import locate_oos_data
from allennlp.predictors import TextClassifierPredictor

# Logger setup.
# log = logging.getLogger(__name__)
# log.debug("Logging is configured.")

# wandb.init(entity="dunkelhaus", project="oos-detect",
# sync_tensorboard=True, reinit=True)

# We've copied the training loop from an earlier example, with updated
# model code, above in the Setup section. We run the training loop to get
# a trained model.


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

# Logits to labels.
for l in preds:
    # print(l)
    argind = np.argmax(l['probs'])
    # print(argind)
    lab = labelmap[argind]
    # print(lab)
    l['predicted'] = lab

#preds['predicted'] = [labelmap[np.argmax(l['probs'])] for l in preds]

# pr = predictor.predictions_to_labeled_instances(test_data[0], preds[0])

for i in range(len(preds)):
    print(f"Instance: {test_data[i]}; Prediction: {preds[i]['predicted']}")
