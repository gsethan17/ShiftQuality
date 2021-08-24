import os
from utils import Dataloader
from model import get_model

train_path = '/home/gsethan/Documents/ShiftQuality/train'
val_path = '/home/gsethan/Documents/ShiftQuality/val'
test_path = '/home/gsethan/Documents/ShiftQuality/test'

n_timewindow = 100
n_feature = 6
latent_size = 10
model_key = 'MLP'


train_path = os.path.join(train_path, str(n_timewindow))
# data load
train_loader = Dataloader(train_path)
X, _ = train_loader[1]
print(X.shape)

n_timewindow = 100
n_feature = 6
latent_size = 10
model_key = 'LSTM'


FC_autoencoder = get_model(model_key, n_timewindow, n_feature, latent_size, show=True)

for i, train_data in enumerate(train_loader):
    print("Training : {} / {}".format(i + 1, len(train_loader)), end="\r")
    train_x, _ = train_data
    print(train_x.shape)
    recon = FC_autoencoder(train_x)
    print(recon.shape)
    break
