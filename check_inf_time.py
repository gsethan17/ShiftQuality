from model import get_shallow_model
from utils import Dataloader
import numpy as np
import os
import configparser
import json
import time

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


    ## path for train results
    train_path = os.path.join(main_dir, 'train')
    if not os.path.isdir(train_path) :
        print('Training has to be done before testing.')
        return -1

    # dir_exist_check([test_dir])

    # GPU limitation
    # limit_gb = int(config['GPU']['LIMIT'])
    # gpu_limit(limit_gb)

    # DATA LOADER
    test_loader = Dataloader(test_data_path, label = True, timewindow=n_timewindow)
    
    model = get_shallow_model(model_key, n_timewindow, n_feature, latent_size, test_layer)

    ## load weights
    weight_path = os.path.join(train_path, 'Best_weights')
    # print(weight_path)
    model.load_weights(weight_path)
    print("---------model loaded----------")
    
    
    times = []
    
    for i, test_data in enumerate(test_loader) :
        # print("Testing : {} / {}".format(i + 1, len(test_loader)), end="\r")
        test_x, test_y, filename = test_data
        
        for j in range(test_x.shape[0]) :
            st = time.time()
            recon = model(test_x[j:j+1])
            ed = time.time()
            
            times.append(ed-st)

    num_mean = np.mean(times)
    std = np.std(times)
    
    print(num_mean, std)


if __name__ == "__main__" :
    base_path = os.path.join(os.getcwd(), "results")

    model = "AE"
    layer = 1
    timewindow = 110

    model_key = model+str(layer)

    model_dir = os.path.join(base_path, model_key, str(timewindow), "0.001_50")
    test_dir = os.path.join(model_dir, 'test')

    test(model_dir, test_dir, layer)
