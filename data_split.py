import os
import glob
import random
import shutil
import pandas as pd
import configparser

# BASIC CONFIGURATION
config = configparser.ConfigParser()
config.read('./config.ini')

## path setup
train_path = config['PATH']['TRAIN']
val_path = config['PATH']['VALIDATION']
test_path = config['PATH']['TEST']

## data shape setup
n_timewindow = int(config['DATA']['N_TIMEWINDOW'])


if not os.path.isdir(train_path) :
    os.mkdir(train_path)
if not os.path.isdir(val_path) :
    os.mkdir(val_path)
if not os.path.isdir(test_path) :
    os.mkdir(test_path)

normal_path = '/home/gsethan/Documents/ShiftQuality/scaled_resample_data'
normal_data_list = glob.glob(os.path.join(normal_path, '*.csv'))

abnormal_path = '/home/gsethan/Documents/ShiftQuality/scaled_anomaly_resample'
abnormal_data_list = glob.glob(os.path.join(abnormal_path, '*.csv'))


for file in normal_data_list :
    df = pd.read_csv(file, index_col=0)
    if len(df) < n_timewindow :
        normal_data_list.remove(file)

for file in abnormal_data_list:
    df = pd.read_csv(file, index_col=0)
    if len(df) < n_timewindow:
        abnormal_data_list.remove(file)

# data split
test_normal_list = random.sample(normal_data_list, len(abnormal_data_list))

for file in test_normal_list :
    normal_data_list.remove(file)

val_normal_list = random.sample(normal_data_list, int(len(normal_data_list)*0.2))
for file in val_normal_list :
    normal_data_list.remove(file)

print("Number of Train data : ", len(normal_data_list))
print("Number of Val data : ", len(val_normal_list))
print("Number of Test data as normal data: ", len(test_normal_list))
print("Number of Test data as abnormal data: ", len(abnormal_data_list))


# file move
## train data
for path in normal_data_list :
    shutil.copyfile(path, os.path.join(train_path, os.path.basename(path)))

## val data
for path in val_normal_list :
    shutil.copyfile(path, os.path.join(val_path, os.path.basename(path)))

## test data
for path in test_normal_list :
    shutil.copyfile(path, os.path.join(test_path, os.path.basename(path)))

for path in abnormal_data_list :
    shutil.copy(path, os.path.join(test_path, "abnormal_"+os.path.basename(path)))

