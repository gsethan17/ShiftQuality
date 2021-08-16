import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

def rmse_loss(recon, origin) :
    n_timewindow = origin.shape[1]
    n_feature = origin.shape[2]

    # calculate rmse
    error = tf.math.subtract(recon, origin)
    error = tf.math.pow(error, 2)
    error = tf.math.reduce_sum(error, axis = 1)
    error = tf.math.reduce_sum(error, axis = 1)
    error = tf.math.divide(error, (n_timewindow*n_feature))
    error = tf.math.sqrt(error)

    # calculate mean of rmse value by batch for train
    error_mean = tf.reduce_mean(error)

    # calculate median and maximum of rmse value by batch for test
    error_array = np.array(error)
    error_median = np.median(error_array)
    error_maximum = np.max(error_array)

    return error_mean, error_median, error_maximum

def get_metric(key) :
    if key == 'rmse' :
        LOSS = rmse_loss

    return LOSS

def get_optimizer(key, lr) :
    if key == 'adam' :
        OPTIMIZER = Adam(learning_rate=lr)

    return OPTIMIZER

def get_model(key, n_timewindow, n_feature, latent_size) :
    if key == 'MLP' :
        model = FC_AE(n_timewindow=n_timewindow, n_feature=n_feature, latent_size=latent_size)


    return model


class Dataloader(Sequence) :
    def __init__(self, path, label = False, timewindow = 100, batch_size=1, shuffle=True):
        self.path = path
        self.label = label
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

    def get_inputs(self, path):
        filename = os.path.basename(path)
        df = pd.read_csv(path, index_col=0)
        col_list = ['HEV_AccelPdlVal', 'HEV_EngSpdVal', \
                   'IEB_StrkDpthPcVal',
                   'WHL_SpdFLVal', \
                   'NTU', 'NAB', 'ntug_SyncFilt']

        if not list(df.columns) == col_list :
            df.columns = col_list

        # delete brake pedal signal
        col_list_rev = ['HEV_AccelPdlVal', 'HEV_EngSpdVal', \
                    # 'IEB_StrkDpthPcVal',
                    'WHL_SpdFLVal', \
                    'NTU', 'NAB', 'ntug_SyncFilt']

        df = df.loc[:, col_list_rev]

        array = np.array(df)

        inputs = []

        n_samples = array.shape[0] - self.timewindow + 1

        for n in range(n_samples) :
            inputs.append(array[n:n+self.timewindow])
        inputs = np.array(inputs)

        return inputs, filename


    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        for i in indices :
            batch_x, filename = self.get_inputs(self.data_list[i])

        if self.label :
            if os.path.basename(self.data_list[i]).split('_')[0] == 'abnormal' :
                batch_y = True
            else:
                batch_y = False
            return batch_x, batch_y, filename
        else :
            return batch_x, batch_x

class FC_Encoder(Model) :
    def __init__(self, latent_size):
        super().__init__()
        self.flat = Flatten()
        self.enc1 = Dense(512, activation = 'relu', name='enc1')
        self.enc2 = Dense(256, activation = 'relu', name='enc2')
        self.enc3 = Dense(128, activation = 'relu', name='enc3')
        self.enc4 = Dense(64, activation = 'relu', name='enc4')
        self.enc5 = Dense(32, activation = 'relu', name='enc5')
        self.enc6 = Dense(latent_size, activation = 'relu', name='enc6')

    def call(self, x):
        x = self.flat(x)
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.enc4(out)
        out = self.enc5(out)
        out = self.enc6(out)

        return out


class FC_Decoder(Model):
    def __init__(self, out_size):
        super().__init__()
        self.dec1 = Dense(32, activation='relu', name='dec1')
        self.dec2 = Dense(64, activation='relu', name='dec2')
        self.dec3 = Dense(128, activation='relu', name='dec3')
        self.dec4 = Dense(256, activation='relu', name='dec4')
        self.dec5 = Dense(512, activation='relu', name='dec5')
        self.dec6 = Dense(out_size, activation='sigmoid', name='dec6')

    def call(self, x):
        out = self.dec1(x)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.dec4(out)
        out = self.dec5(out)
        out = self.dec6(out)

        return out

class FC_AE(Model) :
    def __init__(self, n_timewindow, n_feature, latent_size):
        super().__init__()
        self.n_timewindow = n_timewindow
        self.n_feature = n_feature
        self.latent_size = latent_size
        self.encoder = FC_Encoder(latent_size=self.latent_size)
        self.decoder = FC_Decoder(out_size = self.n_timewindow*self.n_feature)
        self.flat = Flatten()

    def call(self, x):
        batch_size = x.shape[0]
        out = self.encoder(x)
        out = self.decoder(out)
        out = tf.reshape(out, shape=[batch_size, self.n_timewindow, self.n_feature])

        return out

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

def gpu_limit(GB) :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")
    # set the only one GPU and memory limit
    memory_limit = 1024 * GB
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU is not available')

def dir_exist_check(paths) :
    for path in paths :
        if not os.path.isdir(path) :
            os.makedirs(path)

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
