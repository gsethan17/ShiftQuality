import os
import numpy as np
import pandas as pd
import configparser
import json
from model import get_shallow_model
from utils import Dataloader, dir_exist_check, rmse_loss_naive, gpu_limit
from sklearn import metrics


def test(main_dir, test_dir, test_layer) :
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
    # LOSS = get_metric(metric, model_key, USADs)

    # lr = float(json_data['learning_rate'])
    # epochs = int(json_data['epochs'])

    ## path for train results
    train_path = os.path.join(main_dir, 'train')
    if not os.path.isdir(train_path) :
        print('Training has to be done before testing.')
        return -1

    dir_exist_check([test_dir])

    # GPU limitation
    limit_gb = int(config['GPU']['LIMIT'])
    gpu_limit(limit_gb)

    # DATA LOADER
    test_loader = Dataloader(test_data_path, label = True, timewindow=n_timewindow)
    
    model = get_shallow_model(model_key, n_timewindow, n_feature, latent_size, test_layer)

    ## load weights
    weight_path = os.path.join(train_path, 'Best_weights')
    # print(weight_path)
    model.load_weights(weight_path)
    print("---------model loaded----------")


    ## TEST ##
    results = {}
    losses = []

    for i, test_data in enumerate(test_loader) :
        print("Testing : {} / {}".format(i + 1, len(test_loader)), end="\r")
        test_x, test_y, filename = test_data

        recon = model(test_x)

        loss = rmse_loss_naive(recon, test_x)
        losses += list(loss.numpy())
        # print(loss)

        results[filename] = {"label":test_y, "scores":loss}
    '''
    print(len(losses))
    print(len(set(losses)))
    print(set(losses))
    '''

    tpr = []
    fpr = []

    for i, sth in enumerate(set(losses)) :
        print(i, len(set(losses)), end='\r')
        if i % 375 != 0 :
            continue

        TP = 0
        FN = 0
        FP = 0
        TN = 0

        for name in results.keys() :
            preds = results[name]["scores"]

            vote_true = 0
            vote_false = 0

            for pred in preds :
                if pred > sth :
                    vote_true += 1
                else :
                    vote_false += 1

            if vote_true > vote_false :
                results[name]["pred"] = True
            else :
                results[name]["pred"] = False


            if bool(results[name]["label"]) :
                if results[name]["pred"] :
                    TP += 1
                else :
                    FN += 1
            else :
                if results[name]["pred"] :
                    FP += 1
                else :
                    TN += 1

            # print(results[name]["label"],results[name]["pred"], TP, FN, FP, TN)

        tpr.append(TP / (TP + FN))
        fpr.append(FP / (TN + FP))

        # print(len(tpr), len(fpr))

    dic = {'fpr':fpr, 'tpr':tpr}
    df = pd.DataFrame(dic)
    df = df.sort_values(by=["fpr", "tpr"], ascending=[True, True]).copy()
    fpr_ = list(df['fpr'])
    tpr_ = list(df['tpr'])

    print(metrics.auc(fpr_, tpr_))







if __name__ == "__main__" :
    base_path = os.path.join(os.getcwd(), "results")

    model = "AE"
    layer = 2
    timewindow = 70

    model_key = model+str(layer)

    model_dir = os.path.join(base_path, model_key, str(timewindow), "0.001_50")
    test_dir = os.path.join(model_dir, 'test')

    test(model_dir, test_dir, layer)

