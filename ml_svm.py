from sklearn.svm import OneClassSVM
import configparser
import os
from utils import Dataloader, results_analysis, WriteResults, dir_exist_check
import numpy as np
import time

def svm(train_path, val_path, test_path, n_timewindow) :
    ## data shape setup
    # n_timewindow = 140
    n_features = 6

    train_path = os.path.join(train_path, str(n_timewindow))
    test_path = os.path.join(test_path, str(n_timewindow))

    clf = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)

    print(train_path)

    # DATA LOADER
    train_loader = Dataloader(train_path, timewindow=n_timewindow)

    print(len(train_loader))
    for i , train_data in enumerate(train_loader) :
        train_x, _ = train_data
        # print(train_x.shape)
        # print(len(train_x))

        if not n_timewindow == 0 :
            X_train = np.zeros((len(train_x), n_timewindow * n_features))
            for j in range(len(train_x)) :
                train_x_s = np.reshape(train_x[j], (n_timewindow * n_features, ))
                train_x_s = np.expand_dims(train_x_s, axis=0)
                X_train[j] = train_x_s
        else :
            X_train = train_x

        clf.fit(X_train)

    save_path = os.path.join(os.getcwd(), 'results', 'SVM', str(n_timewindow))
    dir_exist_check([save_path])

    test_loader = Dataloader(test_path, label=True, timewindow=n_timewindow)

    test_results = {}
    test_results['filename'] = []
    test_results['label'] = []
    test_results['mean'] = []
    test_results['median'] = []
    test_results['maximum'] = []
    test_results['minimum'] = []

    times = []

    for k, test_data in enumerate(test_loader) :
        print("Testing : {} / {}".format(k + 1, len(test_loader)), end="\r")
        test_x, test_y, filename = test_data

        if not n_timewindow == 0 :
            X_test = np.zeros((len(test_x), n_timewindow * n_features))
            for t in range(len(test_x)) :
                test_x_s = np.reshape(test_x[t], (n_timewindow * n_features, ))
                test_x_s = np.expand_dims(test_x_s, axis=0)
                X_test[t] = test_x_s

        else :
            X_test = test_x

        ## inference time check
        for j in range(X_test.shape[0]) :
            st = time.time()
            y_score = clf.score_samples(X_test[j:j+1])
            ed = time.time()

            times.append(ed-st)

    mean = np.mean(times)
    std = np.std(times)

    print(mean, std)
    
    '''

        # y_pred_train = clf.fit_predict(X_train)
        y_score = clf.score_samples(X_test)

        mean = np.mean(y_score)
        median = np.median(y_score)
        maximum = np.max(y_score)
        minimum = np.min(y_score)

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
    '''

if __name__ == "__main__" :
    config = configparser.ConfigParser()
    config.read('./config.ini')

    train_path = config['PATH']['TRAIN']
    val_path = config['PATH']['VALIDATION']
    test_path = config['PATH']['TEST']

    svm(train_path, val_path, test_path, 140)
    '''
    for i in range(10, 150, 10) :
        svm(train_path, val_path, test_path, i)
    '''
