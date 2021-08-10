from utils import Dataloader, get_lstm_model
import matplotlib.pyplot as plt

# setup
data_path = '/home/gsethan/Documents/ShiftQuality/scaled_resample_data'

n_timewindow = 10
n_feature = 7
n_vector = 32

OPTIMIZER = 'adam'
LOSS = 'mse'
##

# data load
loader = Dataloader(data_path, batch_size=1, timewindow=n_timewindow)
##

# model load & setting
model = get_lstm_model(n_timewindow, n_feature, n_vector)
model.compile(optimizer=OPTIMIZER, loss=LOSS)
##

# model train
history = model.fit(loader, epochs = 2)
##

# results
plt.figure()
plt.plot(history.history['loss'], label = 'Train loss')
plt.legend()
plt.savefig('result.png')
##