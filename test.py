import configparser
import os
import glob
import pandas as pd
from utils import dir_exist_check, get_metric, gpu_limit, Dataloader, get_model


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
    model_version = config['TEST']['VERSION']
    save_path = os.path.join(os.getcwd(), 'results_test', model_key, model_version)
    dir_exist_check([save_path])

    ## metric setup
    metric = config['TRAIN']['METRIC']
    LOSS = get_metric(metric)

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


    # GPU limitation
    gpu_limit(5)


    # DATA LOADER
    test_loader = Dataloader(test_path, label = True)

    # MODEL LOADER
    model = get_model(model_key, n_timewindow, n_feature, latent_size)
    ## load weights
    weight_path = os.path.join(os.getcwd(), 'results_train', model_key, model_version, 'Best_weights')
    model.load_weights(weight_path)

    test_results = {}
    test_results['filename'] = []
    test_results['label'] = []
    test_results['mean'] = []
    test_results['median'] = []
    test_results['maximum'] = []

    for i, test_data in enumerate(test_loader) :
        print("Training : {} / {}".format(i + 1, len(test_loader)), end="\r")
        test_x, test_y, filename = test_data

        recon = model(test_x)
        mean, median, maximum = LOSS(recon, test_x)
        mean = mean.numpy()

        print(mean, median, maximum)
        test_results['filename'].append(filename)
        test_results['label'].append(test_y)
        test_results['mean'].append(mean)
        test_results['median'].append(median)
        test_results['maximum'].append(maximum)

    df = pd.DataFrame(test_results)
    df.to_csv(os.path.join(save_path, 'Results.csv'), index=False)

if __name__ == '__main__' :
    test()








