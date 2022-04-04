import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Flatten, SimpleRNN, GRU, Input, Reshape, Conv1D, MaxPooling1D, UpSampling1D


def get_model(key, n_timewindow, n_feature, latent_size, show=False) :
    model = AE(key=key, n_timewindow=n_timewindow, n_feature=n_feature, latent_size=latent_size, show=show)

    return model


class FC_Encoder(Model) :
    def __init__(self, key, latent_size, show=False):
        super().__init__()
        self.key = key
        self.show = show
        self.flat = Flatten()
        if self.key == 'LSTM' :
            self.enc1 = LSTM(3, return_sequences=True, name='enc1')
        else :
            self.enc1 = Dense(512, activation = 'relu', name='enc1')
        self.enc2 = Dense(256, activation = 'relu', name='enc2')
        self.enc3 = Dense(128, activation = 'relu', name='enc3')
        self.enc4 = Dense(64, activation = 'relu', name='enc4')
        self.enc5 = Dense(32, activation = 'relu', name='enc5')
        self.enc6 = Dense(latent_size, activation = 'relu', name='enc6')

    def show_shape(self, out):
        if self.show :
            print(out.shape)

    def call(self, x):
        if self.key == 'LSTM' :
            out = self.enc1(x)
            self.show_shape(out)
            out = self.flat(out)
            self.show_shape(out)
        else :
            x = self.flat(x)
            self.show_shape(x)
            out = self.enc1(x)
            self.show_shape(out)
        out = self.enc2(out)
        self.show_shape(out)
        out = self.enc3(out)
        self.show_shape(out)
        out = self.enc4(out)
        self.show_shape(out)
        out = self.enc5(out)
        self.show_shape(out)
        out = self.enc6(out)
        self.show_shape(out)

        return out


class FC_Decoder(Model):
    def __init__(self, key, n_timewindow, n_feature, show=False):
        super().__init__()
        self.key = key
        self.show = show
        self.dec1 = Dense(32, activation='relu', name='dec1')
        self.dec2 = Dense(64, activation='relu', name='dec2')
        self.dec3 = Dense(128, activation='relu', name='dec3')
        self.dec4 = Dense(256, activation='relu', name='dec4')
        self.dec5 = Dense(512, activation='relu', name='dec5')
        if self.key == 'LSTM' :
            self.dec6 = LSTM(n_feature, return_sequences=True, name='dec6')
            self.repeat = RepeatVector(n_timewindow, name='repeatvector')
        else :
            self.dec6 = Dense(n_timewindow*n_feature, activation='sigmoid', name='dec6')

    def show_shape(self, out):
        if self.show :
            print(out.shape)

    def call(self, x):
        out = self.dec1(x)
        self.show_shape(out)
        out = self.dec2(out)
        self.show_shape(out)
        out = self.dec3(out)
        self.show_shape(out)
        out = self.dec4(out)
        self.show_shape(out)
        out = self.dec5(out)
        self.show_shape(out)
        if self.key == 'LSTM' :
            out = self.repeat(out)
            self.show_shape(out)
            out = self.dec6(out)
            self.show_shape(out)
        else :
            out = self.dec6(out)
            self.show_shape(out)

        return out

class LSTM_generator(Model):
    def __init__(self, key, n_timewindow, n_feature, show=False):
        super().__init__()
        self.key = key
        self.show = show
        self.gen1 = LSTM(int(n_feature/2), return_sequences=True, name='gen1')
        self.gen2 = LSTM(n_feature, return_sequences=True, name='gen2')

    def show_shape(self, out):
        if self.show :
            print(out.shape)

    def call(self, x):
        out = self.gen1(x)
        self.show_shape(out)
        out = self.gen2(out)
        self.show_shape(out)

class LSTM_discriminator(Model):
    def __init__(self, key, n_timewindow, n_feature, show=False):
        super().__init__()
        self.key = key
        self.show = show
        self.dis1 = LSTM(int(n_feature/2), return_sequences=True, name='dis1')
        self.flat = Flatten(name='flat')
        self.dis2 = Dense(1, activation='sigmoid', name='dis2')

    def show_shape(self, out):
        if self.show :
            print(out.shape)

    def call(self, x):
        out = self.dis1(x)
        self.show_shape(out)
        out = self.flat(out)
        self.show_shape(out)
        out = self.dis2(out)
        self.show_shape(out)

class AE(Model) :
    def __init__(self, key, n_timewindow, n_feature, latent_size, show = False):
        super().__init__()
        self.key = key
        self.show = show
        self.n_timewindow = n_timewindow
        self.n_feature = n_feature
        self.latent_size = latent_size
        if self.key == 'USAD-LSTM' :
            self.encoder = LSTM_Encoder(self.key, self.n_feature, self.latent_size, show=self.show)
            self.decoder = LSTM_Decoder(self.key, self.n_timewindow, self.n_feature, self.latent_size, show=self.show)
            self.decoder2 = LSTM_Decoder(self.key, self.n_timewindow, self.n_feature, self.latent_size, show=self.show)
        else :
            self.encoder = FC_Encoder(key=self.key, latent_size=self.latent_size, show=self.show)
            self.decoder = FC_Decoder(self.key, self.n_timewindow, self.n_feature, show=self.show)
            self.decoder2 = FC_Decoder(self.key, self.n_timewindow, self.n_feature, show=self.show)
            self.flat = Flatten()

    def call(self, x):
        batch_size = x.shape[0]

        if self.key == 'MLP' or self.key == 'LSTM' :
            out = self.encoder(x)
            out = self.decoder(out)
            if self.key == 'MLP' :
                out = tf.reshape(out, shape=[batch_size, self.n_timewindow, self.n_feature])

            return out

        elif self.key == 'USAD' or self.key == 'USAD-LSTM' :
            z = self.encoder(x)
            w1 = self.decoder(z)
            if self.key == 'USAD' :
                w1 = tf.reshape(w1, shape=[batch_size, self.n_timewindow, self.n_feature])
            w2 = self.decoder2(z)
            if self.key == 'USAD' :
                w2 = tf.reshape(w2, shape=[batch_size, self.n_timewindow, self.n_feature])
            w3 = self.decoder2(self.encoder(w1))
            if self.key == 'USAD' :
                w3 = tf.reshape(w3, shape=[batch_size, self.n_timewindow, self.n_feature])

            return w1, w2, w3

        else :
            return -1

class LSTM_Encoder(Model):
    def __init__(self, key, n_feature, latent_size, show=False):
        super().__init__()
        self.key = key
        self.show = show
        self.n_feature = n_feature
        self.latent_size = latent_size
        self.enc1 = LSTM(int(self.n_feature/2), return_sequences=True, name='enc1')
        self.enc2 = LSTM(self.latent_size, name='enc2')

    def show_shape(self, out):
        if self.show:
            print(out.shape)

    def call(self, x):
        out = self.enc1(x)
        self.show_shape(out)
        out = self.enc2(out)
        self.show_shape(out)

        return out


class LSTM_Decoder(Model):
    def __init__(self, key, n_timewindow, n_feature, latent_size, show=False):
        super().__init__()
        self.key = key
        self.show = show
        self.n_timewindow = n_timewindow
        self.n_feature = n_feature
        self.latent_size = latent_size
        self.dec1 = RepeatVector(self.n_timewindow, name='dec1')
        self.dec2 = LSTM(self.latent_size, return_sequences=True, name='dec2')
        self.dec3 = TimeDistributed(Dense(self.n_feature), name='dec3')

    def show_shape(self, out):
        if self.show:
            print(out.shape)

    def call(self, x):
        out = self.dec1(x)
        self.show_shape(out)
        out = self.dec2(out)
        self.show_shape(out)
        out = self.dec3(out)
        self.show_shape(out)

        return out


def get_lstm_model(n_timewindow, n_feature, latent_size) :
    model = Sequential()
    # model.add(LSTM(int(n_feature/2), input_shape=(n_timewindow, n_feature), return_sequences=True))
    model.add(LSTM(latent_size, input_shape=(n_timewindow, n_feature)))
    # model.add(LSTM(latent_size))
    model.add(RepeatVector(n_timewindow))
    # model.add(LSTM(int(n_feature/2), return_sequences=True))
    # model.add(LSTM(n_feature, return_sequences=True))
    model.add(LSTM(latent_size, return_sequences=True))
    model.add(TimeDistributed(Dense(n_feature)))
    print(model.summary())
    return model


def get_shallow_model(model_key, n_timewindow, n_feature, latent_size, layer=1) :
    model = Sequential()
    if model_key == 'AE' :
        model.add(Input(shape=(n_timewindow*n_feature,)))
        model.add(Flatten())
        if layer == 2 :
            model.add(Dense(int((n_timewindow*n_feature)/2), activation ='relu'))
        elif layer == 3 :
            model.add(Dense(int(((n_timewindow*n_feature)/3)*2), activation ='relu'))
            model.add(Dense(int((n_timewindow*n_feature)/3), activation ='relu'))
        elif layer == 4 :
            model.add(Dense(int(((n_timewindow*n_feature)/4)*3), activation ='relu'))
            model.add(Dense(int(((n_timewindow*n_feature)/4)*2), activation ='relu'))
            model.add(Dense(int(((n_timewindow*n_feature)/4)*1), activation ='relu'))

        model.add(Dense(latent_size, activation='relu'))
        if layer == 2 :
            model.add(Dense(int((n_timewindow*n_feature)/2), activation ='relu'))
        elif layer == 3 :
            model.add(Dense(int((n_timewindow*n_feature)/3), activation ='relu'))
            model.add(Dense(int(((n_timewindow*n_feature)/3)*2), activation ='relu'))
        elif layer == 4 :
            model.add(Dense(int(((n_timewindow*n_feature)/4)*1), activation ='relu'))
            model.add(Dense(int(((n_timewindow*n_feature)/4)*2), activation ='relu'))
            model.add(Dense(int(((n_timewindow*n_feature)/4)*3), activation ='relu'))
        model.add(Dense(n_timewindow*n_feature, activation='sigmoid'))
        model.add(Reshape((n_timewindow, n_feature), input_shape=(n_timewindow*n_feature,)))

    if model_key == 'RNN-AE' :
        model.add(Input(shape=(n_timewindow, n_feature)))
        if layer == 2 :
            model.add(SimpleRNN(latent_size, return_sequences=True)) # (10,)
        model.add(SimpleRNN(latent_size)) # (10,)
        model.add(RepeatVector(n_timewindow)) # (n_time, 10)
        model.add(SimpleRNN(latent_size, return_sequences=True)) # (n_time, 10)
        if layer == 2 :
            model.add(SimpleRNN(latent_size, return_sequences=True)) # (10,)
        model.add(TimeDistributed(Dense(n_feature, activation='sigmoid'))) # (n_time, n_feature)

    if model_key == 'LSTM-AE' :
        model.add(Input(shape=(n_timewindow, n_feature)))
        if layer == 2 :
            model.add(LSTM(latent_size, return_sequences=True))
        model.add(LSTM(latent_size))
        model.add(RepeatVector(n_timewindow))
        if layer == 2 :
            model.add(LSTM(latent_size, return_sequences=True))
        model.add(LSTM(latent_size, return_sequences=True))
        model.add(TimeDistributed(Dense(n_feature, activation='sigmoid')))

    if model_key == 'GRU-AE' :
        model.add(Input(shape=(n_timewindow, n_feature)))
        if layer == 2 :
            model.add(GRU(latent_size, return_sequences=True))
        model.add(GRU(latent_size,))
        model.add(RepeatVector(n_timewindow))
        if layer == 2 :
            model.add(GRU(latent_size, return_sequences=True))
        model.add(GRU(latent_size, return_sequences=True))
        model.add(TimeDistributed(Dense(n_feature, activation='sigmoid')))

    if model_key == 'CNN-AE' : 
        model.add(Input(shape=(n_timewindow, n_feature)))
        if layer == 2 :
            model.add(Conv1D(n_feature, 7, activation='relu', padding='same'))
        model.add(Conv1D(1, 7, activation='relu', padding='same'))
        model.add(MaxPooling1D(int(n_timewindow/latent_size), padding='same'))
        model.add(UpSampling1D(int(n_timewindow/latent_size)))
        if layer == 2 :
            model.add(Conv1D(1, 7, activation='relu', padding='same'))
        model.add(Conv1D(n_feature, 7, activation='relu', padding='same'))





    print(model.summary())
    return model

if __name__ == '__main__' :
    model_keys = ['AE', 'RNN-AE', 'LSTM-AE', 'GRU-AE']
    model_keys = ['AE']
    for model_key in model_keys :
        get_shallow_model(model_key, 80, 6, 10, 3)
