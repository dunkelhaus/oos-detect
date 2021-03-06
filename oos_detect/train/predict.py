import numpy as np
from oos_detect.utilities.locate import locate_oos_data
from allennlp.predictors import TextClassifierPredictor


def get_predictions(model, data_reader, data_path):
    predictor = TextClassifierPredictor(
        model=model,
        dataset_reader=data_reader
    )
    data = list(data_reader.read(data_path))

    size = len(data)
    bound = 4000
    preds = []

    if size > bound:
        times = int(size/bound)
        print(f"Set is too big; total size: {size}. "
              f"Batching {times} times.")

        for i in range(times):
            print(f"Lower: {bound*i}, Upper: {bound*(i + 1)}")
            preds += predictor.predict_batch_instance(
                data[bound*i:bound*(i+1)]
            )

        if (size - (bound * times)) > 0:
            print(f"Lower: {bound*times}, Upper: {size}")
            preds += predictor.predict_batch_instance(data[bound*times:])
    else:
        preds = predictor.predict_batch_instance(data)

    labelmap = predictor._model.vocab.get_index_to_token_vocabulary('labels')

    predictions = [
        labelmap[np.argmax(lst['probs'])]
        for lst in preds
    ]
    actuals = [str(i['label'].label) for i in data]
    labels = list(labelmap.values())

    return actuals, predictions, labels
