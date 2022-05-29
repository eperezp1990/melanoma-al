import sys
import json
import os
import time
from multiprocessing import Pool

from utils import create_folds
from execute_AL_fold import evaluate_fold

#models
from models.MobileNet_model import MobileNet_imagenet
from models.DenseNet201_model import DenseNet201_imagenet
from models.InceptionV3_model import InceptionV3_imagenet
from models.Xception_model import Xception_imagenet
from models.ResNet50_model import ResNet50_imagenet
from models.NASNetMobile_model import NASNetMobile_imagenet
from models.VGG16_model import VGG16_imagenet

models = [
    MobileNet_imagenet(),

    DenseNet201_imagenet(),

    InceptionV3_imagenet(),

    Xception_imagenet(),

    NASNetMobile_imagenet(),

    VGG16_imagenet(),

    ResNet50_imagenet(),
]
#end models

# active learning
from AL import RandomSampling, RelevanceProbSampling, LeastConfidentProbSampling, MixLeastRelevanceProbSampling

ALs = [
    RandomSampling,
    RelevanceProbSampling,
    LeastConfidentProbSampling,
    MixLeastRelevanceProbSampling
]
# end active learning

###################### start evaluation ######################

config_file = str(sys.argv[1])

with open(config_file) as json_data:
    configuration = json.load(json_data)

reports_dir = configuration['reportsDir']
gpu = configuration['gpu']

execute_configs = []

for dataset in configuration['datasets']:
    data_folds = create_folds(configuration['folds'], dataset, root='/home/tempdata_activelearning')

    for model in models:
        for optimizer in configuration['optimizers']:
            for al in ALs:
                for version in configuration['generator']:
                    curr_report_dir = os.path.join(reports_dir, dataset['name'], model['name'], \
                        optimizer, al.__name__, 'generator_'+str(version))
                    
                    os.makedirs(curr_report_dir, exist_ok=True)
                    
                    for fold, train_x_f, train_y_f, test_x_f, test_y_f in data_folds:

                        curr_report = os.path.join(curr_report_dir, str(fold) + '_fold.csv')
                        if os.path.exists(curr_report):
                            print('>> done', curr_report)
                            continue
                        execute_configs.append((
                            curr_report,
                            model, 
                            optimizer, 
                            al,
                            train_x_f, 
                            train_y_f, 
                            test_x_f, 
                            test_y_f, 
                            # pre-locate each gpu 
                            gpu[len(execute_configs)%len(gpu)],
                            version,
                            dataset['eval_freq']
                        ))

# execute in each gpu the configurations
print(execute_configs)
for i in range(0, len(execute_configs), len(gpu)):
    with Pool(processes=len(gpu)) as pool:
        pool.starmap(evaluate_fold, execute_configs[i:i+len(gpu)])

            

