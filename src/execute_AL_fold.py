
def evaluate_fold(report_file, map_model, optimizer, active_learning, train_x_f, train_y_f, test_x_f, test_y_f, \
    gpu=0, version=2, eval_freq=1, save_model_path = None):
    
    print('>> start', report_file)

    from Evaluation import Evaluate
    from utils import SGDReduceScheduler

    import time
    import os
    import pandas as pd
    import tensorflow as tf
    from keras import backend as k
    import numpy as np
    np.random.seed(7)

    epochs = 150
    epochs_pre = 20
    batch_size = 8
    transfer = 'pretrained' in map_model

    train_x, train_y = np.load(train_x_f), np.load(train_y_f)
    test_x, test_y = np.load(test_x_f), np.load(test_y_f)

    best_matt = -2

    with tf.device('/gpu:' + str(gpu)):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        try:
            k.set_session(sess)
        except:
            print('No session available')

        model, base_model = map_model['model'](224, 224, 2)

    AL = active_learning(model, train_x, train_y, batch_size, version=version)
    # evaluation metrics
    evaluation = Evaluate(model, train_x, train_y, test_x, test_y)
    # callbacks
    sgd_reducer = None
    if optimizer == 'sgd':
        sgd_reducer = SGDReduceScheduler(model, rate=0.2, epochs=10)

    df = None
    metrics = None

    # review data
    review_path = report_file + '.rank'
    review_rank = {}

    for e in range(epochs + epochs_pre) if transfer else range(epochs):

        print(map_model['name'], e)
        # transfer learning
        if e == 0:
            if transfer:
                print('>> freezing layers')
                countbase = len(base_model.layers)
                for layer in model.layers[:countbase]:
                    layer.trainable = False
            print('>> compile')
            with tf.device('/gpu:' + str(gpu)):
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        if transfer and e == epochs_pre:
            print('>> unfreezing layers')
            countbase = len(base_model.layers)
            for layer in model.layers[:countbase]:
                layer.trainable = True
            print('>> compile')
            with tf.device('/gpu:' + str(gpu)):
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            # update model
            if optimizer == 'sgd':
                sgd_reducer = SGDReduceScheduler(model, rate=0.2, epochs=10)
        # end transfer learning

        # training process
        batch_count = 0
        for batch_x, batch_y in AL.generator():
            # train
            with tf.device('/gpu:' + str(gpu)):
                model.train_on_batch(batch_x, batch_y)
                
                if batch_count % eval_freq == 0:
                    # evaluate
                    logs = {'batch': batch_count, **evaluation.on_batch_end()}
                    # save evaluations
                    if metrics is None:
                        metrics = [m for m in logs]
                        df = pd.DataFrame(columns=metrics)
                    df = df.append(logs, ignore_index=True)
                    df.to_csv(report_file, index=False)
                    if save_model_path is not None and best_matt < logs['test_matthews_corrcoef']:
                        best_matt = logs['test_matthews_corrcoef']
                        model.save(
                            os.path.join(
                                save_model_path,
                                '{model}_{recall:.3f}_{matt:.3f}_{epoch:03d}'.format(**{
                                    'model': map_model['name'], 
                                    'recall': logs['test_recall_label_1'],
                                    'matt': logs['test_matthews_corrcoef'], 
                                    'epoch': e
                                    }) + '.hfd5'
                            )
                        )

            batch_count += 1
            # end batch

        # save ranking for review
        if e > 0:
            for i in AL.instances_ranking:
                if i[0] not in review_rank:
                    review_rank[i[0]] = [i[1]]
                else:
                    review_rank[i[0]] = review_rank[i[0]] + [i[1]]
            
            df_review = pd.DataFrame( columns=['index_name'] + [ 'f_'+str(i) for i in range(e) ] )

            for i in review_rank:
                df_review = df_review.append({'index_name': i, **{'f_'+str(j): review_rank[i][j] for j in range(len(review_rank[i])) }}, ignore_index=True)

            df_review.to_csv(review_path, index=False)

        # end batches
        print('AL evaluation')
        start_time = time.time()

        AL.doEvaluation()
        
        epoch_time = time.time() - start_time
        print('>> epoch time', time.time() - start_time)

        if optimizer == 'sgd':
            sgd_reducer.on_epoch_end()
        
        df.to_csv(report_file, index=False)

        time.sleep(1)
        # end 1 epoch
