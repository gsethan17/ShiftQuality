import os
import numpy as np
import json
import configparser
from utils import get_metric, dir_exist_check, gpu_limit, Dataloader, results_analysis, WriteResults
from model import get_model, get_lstm_model

def get_an_score(test_x) :
    recon = model(test_x)
    mean, median, maximum, minimum = LOSS(recon, test_x)
    mean = mean.numpy()

    return mean, median, maximum, minimum

def get_an_score_usad(test_x, alpha) :
    w1, w2, w3 = model(test_x)
    mean, median, maximum, minimum = LOSS(step=3, recon=w1, rerecon=w3, origin=test_x, a = alpha)

    return mean, median, maximum, minimum

def test_step_usad(test_path) :
    as_ = np.arange(0, 1.1, 0.1)

    for a in as_ :
        sub_test_path = os.path.join(test_path, str(a))
        dir_exist_check([sub_test_path])

        test_step(sub_test_path, model_key = 'USAD', alpha = a)


def test_step(test_path, model_key = 'NA', alpha = 1) :
    # Test inference
    test_results = {}
    test_results['filename'] = []
    test_results['label'] = []
    test_results['mean'] = []
    test_results['median'] = []
    test_results['maximum'] = []
    test_results['minimum'] = []

    for i, test_data in enumerate(test_loader) :
        print("Testing : {} / {}".format(i + 1, len(test_loader)), end="\r")
        test_x, test_y, filename = test_data



        # recon = model(test_x)
        if model_key == 'USAD' :
            mean, median, maximum, minimum = get_an_score_usad(test_x, alpha)
        else :
            mean, median, maximum, minimum = get_an_score(test_x)
        # mean = mean.numpy()

        test_results['filename'].append(filename)
        test_results['label'].append(test_y)
        test_results['mean'].append(mean)
        test_results['median'].append(median)
        test_results['maximum'].append(maximum)
        test_results['minimum'].append(minimum)

    analysis = results_analysis(test_results)
    analyzed_results = analysis.get_metric()
    # print(analyzed_results)

    WriteResults(test_results, analyzed_results, test_path)



def test(main_dir, test_path) :
    global model
    global LOSS
    global test_loader

    # basic configuration
    config = configparser.ConfigParser()
    config.read('./config.ini')

    # parameters setup
    with open(os.path.join(main_dir, 'setup.json'), 'r') as f :
        json_data = json.load(f)
        #print(json_data.keys())

    ## data shape setup
    n_timewindow = int(json_data['n_timewindow'])
    n_feature = int(json_data['n_feature'])
    latent_size = int(json_data['n_latent'])

    ## data path
    test_data_path = config['PATH']['TEST']
    test_data_path = os.path.join(test_data_path, str(n_timewindow))

    ## model setup
    model_key = str(json_data['model'])
    USADs = ['USAD', 'USAD-LSTM']

    ## metric setup
    metric = json_data['metric']
    LOSS = get_metric(metric, model_key, USADs)

    # lr = float(json_data['learning_rate'])
    # epochs = int(json_data['epochs'])

    ## path for train results
    train_path = os.path.join(main_dir, 'train')
    if not os.path.isdir(train_path) :
        print('Training has to be done before testing.')
        return -1

    dir_exist_check([test_path])

    # GPU limitation
    limit_gb = int(config['GPU']['LIMIT'])
    gpu_limit(limit_gb)

    # DATA LOADER
    test_loader = Dataloader(test_data_path, label = True, timewindow=n_timewindow)

    # MODEL LOADER
    # print(model_key, n_timewindow, n_feature, latent_size)
    if model_key == 'LSTM-AE' :
        model = get_lstm_model(n_timewindow, n_feature, latent_size)
    else :
        model = get_model(model_key, n_timewindow, n_feature, latent_size)
    ## load weights
    weight_path = os.path.join(train_path, 'Best_weights')
    # print(weight_path)
    model.load_weights(weight_path)

    if model_key in USADs :
        test_step_usad(test_path)

    else :
        test_step(test_path)



def make_test_results() :
   base_dir = os.path.join(os.getcwd(), 'results')

   if not os.path.isdir(base_dir) :
       print('There are no results that have completed the training.')
       return -1

   model_keys = os.listdir(base_dir)
   # print(model_keys)

   for model_key in model_keys :
       print(model_key)
       # MAD_GAN skip
       if not model_key in ['MLP'] :
           continue

       model_dir = os.path.join(base_dir, model_key)

       timewindows = os.listdir(model_dir)

       for timewindow in timewindows :
           timewindow_dir = os.path.join(model_dir, timewindow)

           params = os.listdir(timewindow_dir)

           for param in params :
               main_dir = os.path.join(timewindow_dir, param)
               test_dir = os.path.join(main_dir, 'test')

               print(test_dir)

               if os.path.isdir(test_dir) :
                   print("{} {} {} is already done.".format(model_key,
                                                               timewindow,
                                                               param))
                   continue
               else :
                   test(main_dir, test_dir)

               # return -1



if __name__ == '__main__' :
    make_test_results()
    print("DONE")
