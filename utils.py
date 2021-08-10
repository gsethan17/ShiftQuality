import numpy as np
import pandas as pd
import os
import glob
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout


class Dataloader(Sequence) :
    def __init__(self, path, timewindow = 10, batch_size=64, shuffle=True):
        self.path = path
        self.data_list = glob.glob(os.path.join(self.path, '*.csv'))
        self.timewindow = timewindow
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data_list)) / float(self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))

        if self.shuffle :
            np.random.shuffle(self.indices)

    def get_lstm_inputs(self, path):
        df = pd.read_csv(path, index_col=0)
        array = np.array(df)

        inputs = []

        n_samples = array.shape[0] - self.timewindow + 1

        for n in range(n_samples) :
            inputs.append(array[n:n+self.timewindow])
        inputs = np.array(inputs)

        return inputs


    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        for i in indices :
            batch_x = self.get_lstm_inputs(self.data_list[i])

        return batch_x, batch_x

def get_lstm_model(n_timewindow, n_feature, n_vector) :
    model = Sequential()
    model.add(LSTM(n_vector, input_shape=(n_timewindow, n_feature)))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(n_timewindow))
    model.add(LSTM(n_vector, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(n_feature)))
    print(model.summary())
    return model

if __name__ == '__main__' :
    data_path = '/home/gsethan/Documents/ShiftQuality/scaled_resample_data'

    n_timewindow = 10
    n_feature = 7
    n_vector = 32

    loader = Dataloader(data_path, batch_size=1, timewindow=n_timewindow)
    # print(loader.path)
    # print(len(loader.data_list))
    # print(len(loader))
    # print(loader[0])

    model = get_lstm_model(n_timewindow, n_feature, n_vector)
    model.compile(optimizer='adam', loss='mae')

    history = model.fit(loader, epochs = 2)

    plt.figure()
    plt.plot(history.history['loss'], label = 'Train loss')
    plt.legend()
    plt.savefig('result.png')
