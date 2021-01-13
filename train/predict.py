import numpy as np
from utilities.locate import locate_oos_data
from allennlp.predictors import TextClassifierPredictor


def get_test_predictions(model, dataset_reader):
    predictor = TextClassifierPredictor(
        model=model,
        dataset_reader=dataset_reader
    )

    test_data = list(
        dataset_reader.read(
            locate_oos_data()/"data_full_test.json"
        )
    )

    preds = predictor.predict_batch_instance(test_data)
    labelmap = predictor._model.vocab.get_index_to_token_vocabulary('labels')

    predictions = [labelmap[np.argmax(l['probs'])] for l in preds]
    actuals = [str(i['label'].label) for i in test_data]
    labels = list(labelmap.values())

    return actuals, predictions, labels
