import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Flatten


def get_model(key, n_timewindow, n_feature, latent_size) :
    if key == 'MLP' :
        model = FC_AE(n_timewindow=n_timewindow, n_feature=n_feature, latent_size=latent_size)


    return model


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