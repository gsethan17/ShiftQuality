import os
import glob
import random
import shutil
import pandas as pd
import configparser
from utils import dir_exist_check

def split_data(compression_rate) :

    # BASIC CONFIGURATION
    config = configparser.ConfigParser()
    config.read('./config.ini')

    ## path setup
    train_path = config['PATH']['TRAIN']
    val_path = config['PATH']['VALIDATION']
    test_path = config['PATH']['TEST']

    ## data shape setup
    n_timewindow = int(config['DATA']['N_TIMEWINDOW'])

    if compression_rate == 1.0 :
        train_path = os.path.join(train_path, str(n_timewindow))
        val_path = os.path.join(val_path, str(n_timewindow))
        test_path = os.path.join(test_path, str(n_timewindow))

    else :
        train_path = os.path.join(train_path, "use_{}".format(compression_rate), str(n_timewindow))
        val_path = os.path.join(val_path, "use_{}".format(compression_rate), str(n_timewindow))
        test_path = os.path.join(test_path, "use_{}".format(compression_rate), str(n_timewindow))

    dir_exist_check([train_path, val_path, test_path])

    if not os.path.isdir(train_path) :
        os.mkdir(train_path)
    if not os.path.isdir(val_path) :
        os.mkdir(val_path)
    if not os.path.isdir(test_path) :
        os.mkdir(test_path)


    normal_path = '/home/gsethan/Documents/ShiftQuality/scaled_resample_data'
    normal_data_base = glob.glob(os.path.join(normal_path, '*.csv'))
    normal_data_list = []

    abnormal_path = '/home/gsethan/Documents/ShiftQuality/scaled_anomaly_resample'
    abnormal_data_base = glob.glob(os.path.join(abnormal_path, '*.csv'))
    abnormal_data_list = []


    for file in normal_data_base :
        df = pd.read_csv(file, index_col=0)
        if not len(df) < n_timewindow:
            normal_data_list.append(file)

    for file in abnormal_data_base:
        df = pd.read_csv(file, index_col=0)
        if not len(df) < n_timewindow:
            abnormal_data_list.append(file)

    # data split
    test_normal_list = random.sample(normal_data_list, len(abnormal_data_list))

    for file in test_normal_list :
        normal_data_list.remove(file)

    train_normal_list = random.sample(normal_data_list, int(len(normal_data_list)*compression_rate))

    val_normal_list = random.sample(train_normal_list, int(len(train_normal_list)*0.2))
    for file in val_normal_list :
        train_normal_list.remove(file)

    print("Number of Train data : ", len(train_normal_list))
    print("Number of Val data : ", len(val_normal_list))
    print("Number of Test data as normal data: ", len(test_normal_list))
    print("Number of Test data as abnormal data: ", len(abnormal_data_list))


    # file move
    ## train data
    for path in train_normal_list :
        shutil.copyfile(path, os.path.join(train_path, os.path.basename(path)))

    ## val data
    for path in val_normal_list :
        shutil.copyfile(path, os.path.join(val_path, os.path.basename(path)))

    ## test data
    for path in test_normal_list :
        shutil.copyfile(path, os.path.join(test_path, os.path.basename(path)))

    for path in abnormal_data_list :
        shutil.copy(path, os.path.join(test_path, "abnormal_"+os.path.basename(path)))

if __name__ == "__main__" :

    split_data(1.0)

    '''
    for i in range(1, 10, 1) :
        use_rate = round(i*0.1, 2)
        print(use_rate)
        split_data(use_rate)
    '''
