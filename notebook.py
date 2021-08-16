from utils import Dataloader, FC_AE

train_path = '/home/gsethan/Documents/ShiftQuality/train'
val_path = '/home/gsethan/Documents/ShiftQuality/val'
test_path = '/home/gsethan/Documents/ShiftQuality/test'

# data load
train_loader = Dataloader(train_path)

X, _ = train_loader[1]
print(X.shape)

test_loader = Dataloader(test_path, label=True)
X, Y = test_loader[1]
print(X.shape)
print(Y)

n_timewindow = 100
n_feature = 6
latent_size = 10
FC_autoencoder = FC_AE(n_timewindow, n_feature, latent_size)

for i, train_data in enumerate(train_loader):
    print("Training : {} / {}".format(i + 1, len(train_loader)), end="\r")
    train_x, _ = train_data
    print(train_x.shape)
    recon = FC_autoencoder(train_x)
    print(recon.shape)