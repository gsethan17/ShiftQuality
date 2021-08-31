import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Flatten


def get_model(key, n_timewindow, n_feature, latent_size, show=False) :
    model = FC_AE(key=key, n_timewindow=n_timewindow, n_feature=n_feature, latent_size=latent_size, show=show)

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
    def __init__(self, key, n_timewindow, n_feature, show = False):
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

class FC_AE(Model) :
    def __init__(self, key, n_timewindow, n_feature, latent_size, show = False):
        super().__init__()
        self.key = key
        self.show = show
        self.n_timewindow = n_timewindow
        self.n_feature = n_feature
        self.latent_size = latent_size
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

        elif self.key == 'USAD' :
            z = self.encoder(x)
            w1 = self.decoder(z)
            w1 = tf.reshape(w1, shape=[batch_size, self.n_timewindow, self.n_feature])
            w2 = self.decoder2(z)
            w2 = tf.reshape(w2, shape=[batch_size, self.n_timewindow, self.n_feature])
            w3 = self.decoder2(self.encoder(w1))
            w3 = tf.reshape(w3, shape=[batch_size, self.n_timewindow, self.n_feature])

            return w1, w2, w3

        else :
            return -1

def get_lstm_model(n_timewindow, n_feature, latent_size) :
    model = Sequential()
    model.add(LSTM(latent_size, input_shape=(n_timewindow, n_feature)))
    model.add(RepeatVector(n_timewindow))
    model.add(LSTM(latent_size, return_sequences=True))
    model.add(TimeDistributed(Dense(n_feature)))
    print(model.summary())
    return model