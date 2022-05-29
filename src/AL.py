import numpy as np
import math
from sklearn.utils import shuffle

def ordered_generator(data, batch_size):
    d = [i for i in range(data.shape[0])]
    while True:
        size = batch_size if len(d) >= batch_size else len(d)
        yield data[d[:size]]
        d = d[size:]

def random_generator(datax, datay, batch_size):
    d = range(datax.shape[0])
    d = shuffle(d)
    while True:
        size = batch_size if len(d) >= batch_size else len(d)
        yield datax[d[:size]], datay[d[:size]]
        d = d[size:]


class ActiveLearning:
    
    def __init__(self, model, datax, datay, batch_size, version=2):
        self.model = model
        self.datax = datax
        self.datay = datay
        self.batch_size = batch_size
        self.max = True

        # {id_inst, folds}
        self.instances_ranking = []

        self.version = version
        if version == 2:
            self.generator = self.generator_v2
        elif version == 1:
            self.generator = self.generator_v1

        unique, counts = np.unique(datay, return_counts=True)
        self.classes = dict(zip(unique, counts))
        for i in self.classes:
            self.classes[i] = int((self.classes[i]/self.datax.shape[0])*batch_size)
        
        self.weights = np.ones(datax.shape[0])

    # equal distribution of samples peer batch
    def generator_v1(self):
        bag = [ (i, self.datay[i], self.weights[i]) for i in range(self.datax.shape[0])]
        
        # order by weight
        bag.sort(key= lambda v: v[2], reverse=self.max)
        
        # list by each class
        by_class = [ 
            ordered_generator(
                np.asarray([b[0] for b in bag if b[1] == c]), 
                self.classes[c]
            ) for c in self.classes
        ]

        for _ in range(math.ceil(len(bag)//self.batch_size)):
            # select indexes 
            selected = [next(i) for i in by_class]
            selected = np.concatenate(selected)
            yield self.datax[selected], self.datay[selected]

    # get better results, pick max or min without balance categories
    def generator_v2(self):
        bag = [ (i, self.datay[i], self.weights[i]) for i in range(self.datax.shape[0])]
        
        # order by weight
        bag.sort(key= lambda v: v[2], reverse=self.max)
        
        # indexe, ranking, class
        self.instances_ranking = [ (bag[i][0], i, bag[i][1]) for i in range(len(bag)) ]

        indexes = [i[0] for i in bag]
        while len(indexes) > 0:
            size = self.batch_size if len(indexes) >= self.batch_size else len(indexes)
            yield self.datax[indexes[:size]], self.datay[indexes[:size]]
            indexes = indexes[size:]

# -----------------------------------------
class RandomSampling(ActiveLearning):

    def __init__(self, model, datax, datay, batch_size, version):
        ActiveLearning.__init__(self, model, datax, datay, batch_size, version)

    def generator(self):
        d = range(self.datax.shape[0])
        d = shuffle(d)
        while len(d) > 0:
            size = self.batch_size if len(d) >= self.batch_size else len(d)
            yield self.datax[d[:size]], self.datay[d[:size]]
            d = d[size:]

    def doEvaluation(self):
        self.weights = self.weights

# -----------------------------------------
# based on loss
class UncertaintySampling(ActiveLearning):

    def __init__(self, model, datax, datay, batch_size, version):
        ActiveLearning.__init__(self, model, datax, datay, batch_size, version)


    def doEvaluation(self):
        self.weights = np.asarray([
            self.model.test_on_batch(self.datax[i:i+1], self.datay[i:i+1])[0] for i in range(self.datax.shape[0])
        ])    

class RelevanceSampling(UncertaintySampling):

    def __init__(self, model, datax, datay, batch_size, version):
        UncertaintySampling.__init__(self, model, datax, datay, batch_size, version)

        self.max = False


class LeastConfidentSampling(UncertaintySampling):

    def __init__(self, model, datax, datay, batch_size, version):
        UncertaintySampling.__init__(self, model, datax, datay, batch_size, version)

        self.max = True

# based on probabilities
class UncertaintyProbSampling(ActiveLearning):

    def __init__(self, model, datax, datay, batch_size, version):
        ActiveLearning.__init__(self, model, datax, datay, batch_size, version)

    def doEvaluation(self):
        self.weights = np.abs(0.5 - self.model.predict(self.datax, batch_size=self.batch_size).flatten())  

class RelevanceProbSampling(UncertaintyProbSampling):

    def __init__(self, model, datax, datay, batch_size, version):
        UncertaintySampling.__init__(self, model, datax, datay, batch_size, version)

        self.max = True

class LeastConfidentProbSampling(UncertaintyProbSampling):

    def __init__(self, model, datax, datay, batch_size, version):
        UncertaintySampling.__init__(self, model, datax, datay, batch_size, version)

        self.max = False

# ---------------------------------------------------------------------------------------

class MixLeastRelevanceProbSampling:

    def __init__(self, model, datax, datay, batch_size, version=2):
        self.model = model
        self.datax = datax
        self.datay = datay
        self.batch_size = batch_size
        self.max = True

        # {id_inst, folds}
        self.instances_ranking = []

        self.version = version
        if version == 2:
            self.generator = self.generator_v2
        elif version == 1:
            self.generator = self.generator_v1

        unique, counts = np.unique(datay, return_counts=True)
        self.classes = dict(zip(unique, counts))
        for i in self.classes:
            self.classes[i] = int((self.classes[i]/self.datax.shape[0])*batch_size)
        
        self.weights = np.ones(datax.shape[0])


    def doEvaluation(self):
        self.weights = np.abs(0.5 - self.model.predict(self.datax, batch_size=self.batch_size).flatten())  


    # equal distribution of samples peer batch
    def generator_v1(self):
        bag = [ (i, self.datay[i], self.weights[i]) for i in range(self.datax.shape[0])]
        
        # order by weight
        bag.sort(key= lambda v: v[2], reverse=self.max)
        
        # list by each class
        by_class = [ 
            ordered_generator(
                np.asarray([b[0] for b in bag if b[1] == c]), 
                self.classes[c]
            ) for c in self.classes
        ]

        for _ in range(math.ceil(len(bag)//self.batch_size)):
            # select indexes 
            selected = [next(i) for i in by_class]
            selected = np.concatenate(selected)
            yield self.datax[selected], self.datay[selected]


    # get better results, pick max or min without balance categories
    def generator_v2(self):
        bag = [ (i, self.datay[i], self.weights[i]) for i in range(self.datax.shape[0])]
        
        # order by weight
        bag.sort(key= lambda v: v[2], reverse=self.max)
        
        # indexe, ranking, class
        self.instances_ranking = [ (bag[i][0], i, bag[i][1]) for i in range(len(bag)) ]

        indexes = [i[0] for i in bag]
        while len(indexes) > 0:
            size = self.batch_size if len(indexes) >= self.batch_size else len(indexes)
            yield self.datax[indexes[:size]], self.datay[indexes[:size]]
            indexes = indexes[size:]

        # review new query strategy, change value after each epoch
        self.max = not self.max