import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

timeseries = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       [0.1**3, 0.2**3, 0.3**3, 0.4**3, 0.5**3, 0.6**3, 0.7**3, 0.8**3, 0.9**3]]).transpose()

timesteps = timeseries.shape[0]
n_features = timeseries.shape[1]

print(timeseries.shape)
print(timesteps, n_features)

timesteps = 1
X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)
print(X)


n_features = 2
X = np.array(X)
X = X.reshape(X.shape[0], timesteps, n_features)

print(X.shape)
print(X)