from sklearn.svm import OneClassSVM
import configparser
import os
from utils import Dataloader
import numpy as np

config = configparser.ConfigParser()
config.read('./config.ini')

train_path = config['PATH']['TRAIN']
val_path = config['PATH']['VALIDATION']
test_path = config['PATH']['TEST']

## data shape setup
n_timewindow = 50

train_path = os.path.join(train_path, str(n_timewindow))
test_path = os.path.join(test_path, str(n_timewindow))

clf = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)

# data shape setup
n_timewindow = 0
n_features = 6

print(train_path)

# DATA LOADER
train_loader = Dataloader(train_path, timewindow=n_timewindow)

print(len(train_loader))
for i , train_data in enumerate(train_loader) :
    train_x, _ = train_data
    print(train_x.shape)
    print(len(train_x))

    if not n_timewindow == 0 :
        X_train = np.zeros((len(train_x), n_timewindow * n_features))
        for j in range(len(train_x)) :
            train_x_s = np.reshape(train_x[j], (n_timewindow * n_features, ))
            train_x_s = np.expand_dims(train_x_s, axis=0)
            X_train[j] = train_x_s
    else :
        X_train = train_x

    clf.fit(X_train)


test_loader = Dataloader(test_path, n_timewindow=n_timewindow)

for k, test_data in enumerate(test_loader) :

    # y_pred_train = clf.fit_predict(X_train)
    y_score = clf.score_samples(test_data)

thresh = np.quantile(y_score, 0.03)
print(thresh)

index = np.where(y_score<=thresh)

print(thresh)
print(y_pred_train.shape)
print(y_pred_train)
