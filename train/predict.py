import numpy as np
from utilities.locate import locate_oos_data
from allennlp.predictors import TextClassifierPredictor


def get_predictions(model, dataset_reader, data):
    predictor = TextClassifierPredictor(
        model=model,
        dataset_reader=dataset_reader
    )

    preds = predictor.predict_batch_instance(data)
    labelmap = predictor._model.vocab.get_index_to_token_vocabulary('labels')

    predictions = [labelmap[np.argmax(l['probs'])] for l in preds]
    actuals = [str(i['label'].label) for i in data]
    labels = list(labelmap.values())

    return actuals, predictions, labels
