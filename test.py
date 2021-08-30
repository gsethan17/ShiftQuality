import configparser
import os
import glob
import pandas as pd
from utils import dir_exist_check, get_metric, gpu_limit, Dataloader, results_analysis, WriteResults
from model import get_model

def test() :
    # BASIC CONFIGURATION
    config = configparser.ConfigParser()
    config.read('./config.ini')

    ## path setup
    test_path = config['PATH']['TEST']

    ## data shape setup
    n_timewindow = int(config['DATA']['N_TIMEWINDOW'])
    n_feature = int(config['DATA']['N_FEATURE'])
    latent_size = int(config['DATA']['LATENT_SIZE'])

    test_path = os.path.join(test_path, str(n_timewindow))

    ## model setup
    model_key = config['MODEL']['KEY']
    # model_version = config['TEST']['VERSION']

    ## metric setup
    metric = config['TRAIN']['METRIC']
    LOSS = get_metric(metric)

    learning_rate = float(config['TRAIN']['LEARNING_RATE'])
    epochs = int(config['TRAIN']['EPOCHS'])

    # path for train results
    train_path = os.path.join(os.getcwd(), 'results', model_key, str(n_timewindow),
                             str(learning_rate) + '_' + str(epochs))
    if not os.path.isdir(train_path) :
        print('Training has to be dene before testing.')
        return -1

    # save path setup
    save_path = os.path.join(train_path, 'test')
    dir_exist_check([save_path])
    '''
    # save parameters
    f = open(os.path.join(save_path, "setting.txt"), 'w')
    settings = "Train model : {}\n" \
               "The size of time window : {}\n" \
               "The number of features : {}\n" \
               "The size of latent vector : {}\n" \
               "The metric : {}\n".format(
        model_key, n_timewindow, n_feature, latent_size,
        metric
    )
    f.write(settings)
    f.close()
    '''


    # GPU limitation
    limit_gb = int(config['GPU']['LIMIT'])
    gpu_limit(limit_gb)


    # DATA LOADER
    test_loader = Dataloader(test_path, label = True, timewindow=n_timewindow)

    # MODEL LOADER
    model = get_model(model_key, n_timewindow, n_feature, latent_size)
    ## load weights
    weight_path = os.path.join(train_path, 'train', 'Best_weights')
    model.load_weights(weight_path)

    test_results = {}
    test_results['filename'] = []
    test_results['label'] = []
    test_results['mean'] = []
    test_results['median'] = []
    test_results['maximum'] = []
    test_results['minimum'] = []

    for i, test_data in enumerate(test_loader) :
        print("Training : {} / {}".format(i + 1, len(test_loader)), end="\r")
        test_x, test_y, filename = test_data

        recon = model(test_x)
        mean, median, maximum, minimum = LOSS(recon, test_x)
        mean = mean.numpy()

        test_results['filename'].append(filename)
        test_results['label'].append(test_y)
        test_results['mean'].append(mean)
        test_results['median'].append(median)
        test_results['maximum'].append(maximum)
        test_results['minimum'].append(minimum)

    analysis = results_analysis(test_results)
    analyzed_results = analysis.get_metric()
    # print(analyzed_results)

    WriteResults(test_results, analyzed_results, save_path)




if __name__ == '__main__' :
    test()








