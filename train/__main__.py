import logging.config
from allennlp.training.util import evaluate
from configs.log.log_conf import LOGGING_CONFIG
from oos_eval.train.loop import run_training_loop
from oos_detect.utilities.locate import locate_oos_data

# --- Universal logger setup - startup task ---
logging.config.dictConfig(LOGGING_CONFIG)


# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop()

# Now we can evaluate the model on a new dataset.
test_data = dataset_reader.read(locate_oos_data()/"data_full_test.json")
test_data.index_with(model.vocab)
data_loader = DataLoader(test_data, batch_size=8)

results = evaluate(model, data_loader)
print(results)
