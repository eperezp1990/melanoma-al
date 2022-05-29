import numpy as np
import uuid
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from keras import backend as K

from keras.applications.inception_v3 import preprocess_input

from balance import image_aug_balance

def create_folds(num, dataset, root=None):
    np.random.seed(7)

    folds_path = './folds_files' if root is None else root
    os.makedirs(folds_path, exist_ok=True)

    X, Y = np.load(dataset['x']), np.load(dataset['y'])

    X = preprocess_input(X)

    kfold = StratifiedKFold(n_splits=num, shuffle=True, random_state=7)

    folds = []

    for train, test in kfold.split(X, Y):

        name_train = '%s-fold-%s-train' % (dataset['name'], str(len(folds)))
        train_x_f, train_y_f =  os.path.join(folds_path, name_train + '_X.npy') , os.path.join(folds_path, name_train + '_Y.npy')

        name_test = '%s-fold-%s-test' % (dataset['name'], str(len(folds)))
        test_x_f, test_y_f =  os.path.join(folds_path, name_test + '_X.npy') , os.path.join(folds_path, name_test + '_Y.npy')

        if os.path.exists(train_x_f) is False:
            train_x, train_y = X[train], Y[train]
            train_x, train_y, indices2 = image_aug_balance(train_x,train_y,0)

            np.save(train_x_f, train_x)
            np.save(train_y_f, train_y)
            del train_x
            del train_y

            test_x, test_y = X[test], Y[test]
            
            np.save(test_x_f, test_x)
            np.save(test_y_f, test_y)
            del test_x
            del test_y

        folds.append((len(folds), train_x_f, train_y_f, test_x_f, test_y_f))

    del X
    del Y
    return folds


class SGDReduceScheduler:

    def __init__(self, model, rate=0.2, epochs=10):
        self.model = model
        self.epochs = epochs
        self.rate = rate
        self.count = 0.

    def on_epoch_end(self):
        self.count += 1
        if self.count//self.epochs == self.count/self.epochs:
            K.set_value(
                self.model.optimizer.lr, 
                K.get_value(self.model.optimizer.lr) * self.rate
            )
            print('>> new learning rate', K.get_value(self.model.optimizer.lr))
