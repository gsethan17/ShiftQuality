from utils import Dataloader, gpu_limit, dir_exist_check, get_metric, get_optimizer
from model import get_model
import tensorflow as tf
import configparser
import pandas as pd
import os
import glob

def train() :
    # BASIC CONFIGURATION
    config = configparser.ConfigParser()
    config.read('./config.ini')

    ## path setup
    train_path = config['PATH']['TRAIN']
    val_path = config['PATH']['VALIDATION']

    ## data shape setup
    n_timewindow = int(config['DATA']['N_TIMEWINDOW'])
    n_feature = int(config['DATA']['N_FEATURE'])
    latent_size = int(config['DATA']['LATENT_SIZE'])

    train_path = os.path.join(train_path, str(n_timewindow))
    val_path = os.path.join(val_path, str(n_timewindow))

    ## model setup
    model_key = config['MODEL']['KEY']
    save_dir = os.path.join(os.getcwd(), 'results_train', model_key)
    dir_exist_check([save_dir])

    save_path = os.path.join(save_dir, "{}".format(len(glob.glob(os.path.join(save_dir, '*')))))
    dir_exist_check([save_path])


    ## train setup
    metric = config['TRAIN']['METRIC']
    LOSS = get_metric(metric)

    optimizer = config['TRAIN']['OPTIMIZER']
    learning_rate = float(config['TRAIN']['LEARNING_RATE'])
    OPTIMIZER = get_optimizer(optimizer, learning_rate)

    epochs = int(config['TRAIN']['EPOCHS'])


    # GPU limitation
    limit_gb = int(config['GPU']['LIMIT'])
    gpu_limit(limit_gb)


    # save parameters
    f = open(os.path.join(save_path, "setting.txt"), 'w')
    settings = "Train model : {}\n" \
               "The size of time window : {}\n" \
               "The number of features : {}\n" \
               "The size of latent vector : {}\n" \
               "The metric : {}\n" \
               "The optimizer : {}\n" \
               "The learning rate : {}\n" \
               "The number of epochs : {}".format(
        model_key, n_timewindow, n_feature, latent_size,
        metric, optimizer, learning_rate, epochs
    )
    f.write(settings)
    f.close()




    # DATA LOADER
    train_loader = Dataloader(train_path, timewindow=n_timewindow)
    val_loader = Dataloader(val_path, timewindow=n_timewindow)

    print(len(train_loader), len(val_loader))


    # MODEL LOADER
    model = get_model(model_key, n_timewindow, n_feature, latent_size)
    print("Model is loaded!")

    results = {}
    results['train_loss'] = []
    results['val_loss'] = []

    for epoch in range(epochs) :
        train_loss = []
        for i, train_data in enumerate(train_loader) :
            print("Training : {} / {}".format(i + 1, len(train_loader)), end="\r")
            train_x, _ = train_data
            with tf.GradientTape() as tape :
                recon = model(train_x)
                loss, _, _, _ = LOSS(recon, train_x)
                train_loss.append(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            OPTIMIZER.apply_gradients(zip(gradients, model.trainable_variables))
        print("Training is completed...")
        train_loader.on_epoch_end()

        val_loss = []
        for j, val_data in enumerate(val_loader) :
            print("Validation : {} / {}".format(j + 1, len(val_loader)), end="\r")
            val_x, _ = val_data

            recon = model(val_x)
            loss, _, _, _ = LOSS(recon, val_x)
            val_loss.append(loss)
        val_loader.on_epoch_end()
        print("Validation is completed...")


        # save results
        train_loss_avg = sum(train_loss) / len(train_loss)
        val_loss_avg = sum(val_loss) / len(val_loss)

        model.save_weights(os.path.join(save_path, "{}epoch_weights".format(epoch + 1)))

        if epoch > 0:
            if val_loss_avg.numpy() < min(results['val_loss']):
                print(val_loss_avg.numpy(), min(results['val_loss']))
                # save best weights
                model.save_weights(os.path.join(save_path, "Best_weights".format(epoch +1)))

        results['train_loss'].append(train_loss_avg.numpy())
        results['val_loss'].append(val_loss_avg.numpy())

        print(
            "{:>3} / {:>3} || train_loss:{:8.4f}, val_loss:{:8.4f}".format(
                epoch + 1, epochs,
                results['train_loss'][-1],
                results['val_loss'][-1],))

        # early stop
        if epoch > 5:
            if results['val_loss'][-5] < min(results['val_loss'][-4:]):
                print(results['val_loss'][-5])
                print(min(results['val_loss'][-4:]))
                break

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)


if __name__ == '__main__' :
    train()
