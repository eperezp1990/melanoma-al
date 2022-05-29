import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_auc_score, accuracy_score, \
    average_precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
from keras.callbacks import Callback

from balance import image_aug_balance

def predict_classes(model, x, batch_size=None, verbose=0, steps=None):
    """Generate class predictions for the input samples.

    The input samples are processed batch by batch.

    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: Integer. If unspecified, it will default to 32.
        verbose: verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`.

    # Returns
        A numpy array of class predictions.
    """
    proba = model.predict(x, batch_size=batch_size, verbose=verbose,
                          steps=steps)

    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')

def predict_classes_average(model, x, indices, batch_size=None, verbose=0, steps=None):
    """Generate class predictions for the input samples.

    The input samples are processed batch by batch.

    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: Integer. If unspecified, it will default to 32.
        verbose: verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`.

    # Returns
        A numpy array of class predictions.
    """
    proba = model.predict(x, batch_size=batch_size, verbose=verbose,
                        steps=steps)
    
    proba = np.asarray([np.average(proba[indices[i]], axis=0) for i in indices])

    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')


class Evaluate():

    def __init__(self, model, trainx, trainy, testx, testy):
        self.model = model
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy
        self.testx_aug, self.testy_aug, self.indices = image_aug_balance(self.testx, self.testy, 0)

    def on_batch_end(self):
       logs = {
           **self.evaluate_on(self.trainx, self.trainy, 'train_'),
           **self.evaluate_on(self.testx, self.testy, 'test_'),
           **self.evaluate_on_average(self.testx_aug, self.testy_aug, 'test_aug_', indices=self.indices)
        }
       return logs

    def evaluate_on(self, x, y, prefix):
        y_pred = predict_classes(self.model, x, batch_size=8)

        logs = {}
        logs[prefix+'loss'] = self.model.evaluate(x, y, batch_size=8)[0]
        logs[prefix+'accuracy'] = accuracy_score(y, y_pred)
        logs[prefix+'matthews_corrcoef'] = matthews_corrcoef(y, y_pred)
        
        sen_values = recall_score(y, y_pred, average=None)
        f1_values = f1_score(y, y_pred, average=None)
        for i in range(len(sen_values)):
            logs[prefix+'recall_label_'+str(i)] = sen_values[i]
            logs[prefix+'f1_label_'+str(i)] = f1_values[i]
        return logs

    def evaluate_on_average(self, x, y, prefix, indices):
        y_pred = predict_classes_average(self.model, x, indices, batch_size=8)
        y_true = y[np.asarray(list(indices.keys()))]

        logs = {}
        logs[prefix+'loss'] = self.model.evaluate(x, y, batch_size=8)[0]
        logs[prefix+'accuracy'] = accuracy_score(y_true, y_pred)
        logs[prefix+'matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        sen_values = recall_score(y_true, y_pred, average=None)
        f1_values = f1_score(y_true, y_pred, average=None)
        for i in range(len(sen_values)):
            logs[prefix+'recall_label_'+str(i)] = sen_values[i]
            logs[prefix+'f1_label_'+str(i)] = f1_values[i]
        return logs