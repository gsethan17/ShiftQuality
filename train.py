from utils import Dataloader, gpu_limit, dir_exist_check, get_metric, get_optimizer
from model import get_model, get_lstm_model
import tensorflow as tf
import configparser
import pandas as pd
import os
import json

def train_step_usad(train_x, epoch) :


    with tf.GradientTape(persistent=True) as tape:
        w1, w2, w3 = model(train_x)
        loss1 = LOSS(step=1, recon=w1, rerecon=w3, origin=train_x, n=epoch+1)

    gradients_enc = tape.gradient(loss1, model.encoder.trainable_variables)
    gradients_dec = tape.gradient(loss1, model.decoder.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients_enc, model.encoder.trainable_variables))
    OPTIMIZER.apply_gradients(zip(gradients_dec, model.decoder.trainable_variables))

    with tf.GradientTape(persistent=True) as tape:
        w1, w2, w3 = model(train_x)
        loss2 = LOSS(step=2, recon=w2, rerecon=w3, origin=train_x, n=epoch+1)

    gradients_enc = tape.gradient(loss2, model.encoder.trainable_variables)
    gradients_dec = tape.gradient(loss2, model.decoder2.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients_enc, model.encoder.trainable_variables))
    OPTIMIZER.apply_gradients(zip(gradients_dec, model.decoder2.trainable_variables))

    return loss1, loss2

def val_step_usad(val_x, epoch) :
    w1, w2, w3 = model(val_x)

    loss1 = LOSS(1, w1, w3, val_x, epoch+1)
    loss2 = LOSS(2, w2, w3, val_x, epoch+1)

    return loss1, loss2

def train_step(train_x) :

    with tf.GradientTape() as tape:
        recon = model(train_x)
        loss, _, _, _ = LOSS(recon, train_x)
        # train_loss.append(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def val_step(val_x) :

    recon = model(val_x)
    loss, _, _, _ = LOSS(recon, val_x)

    return loss


def train() :
    global model
    global LOSS
    global OPTIMIZER

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


    ## train setup
    metric = config['TRAIN']['METRIC']
    LOSS = get_metric(metric, model_key)

    optimizer = config['TRAIN']['OPTIMIZER']
    learning_rate = float(config['TRAIN']['LEARNING_RATE'])
    OPTIMIZER = get_optimizer(optimizer, learning_rate)

    epochs = int(config['TRAIN']['EPOCHS'])

    # save path setup
    save_path_base = os.path.join(os.getcwd(), 'results', model_key, str(n_timewindow), str(learning_rate) + '_' + str(epochs))
    save_path = os.path.join(os.getcwd(), 'results', model_key, str(n_timewindow), str(learning_rate) + '_' + str(epochs), 'train')
    dir_exist_check([save_path])
    # os.makedirs(save_path)


    # GPU limitation
    limit_gb = int(config['GPU']['LIMIT'])
    gpu_limit(limit_gb)


    # save parameters
    param = {}
    param['model'] = model_key
    param['n_timewindow'] = n_timewindow
    param['n_feature'] = n_feature
    param['n_latent'] = latent_size
    param['metric'] = metric
    param['optimizer'] = optimizer
    param['learning_rate'] = learning_rate
    param['epochs'] = epochs

    json_save_path = os.path.join(save_path_base, 'setup.json')

    with open(json_save_path, 'w', encoding='utf-8') as make_file :
        json.dump(param, make_file, ensure_ascii=False, indent='\t')


    # DATA LOADER
    train_loader = Dataloader(train_path, timewindow=n_timewindow)
    val_loader = Dataloader(val_path, timewindow=n_timewindow)

    print(len(train_loader), len(val_loader))


    # MODEL LOADER
    if model_key == 'LSTM-AE' :
        model = get_lstm_model(n_timewindow, n_feature, latent_size)
    else :
        model = get_model(model_key, n_timewindow, n_feature, latent_size)
    print("Model is loaded!")

    results = {}
    results['train_loss'] = []
    results['val_loss'] = []
    if model_key == 'USAD' :
        results['train_loss1'] = []
        results['train_loss2'] = []
        results['val_loss1'] = []
        results['val_loss2'] = []

    for epoch in range(epochs) :
        train_loss = []
        if model_key == 'USAD' :
            train_loss2 = []
        for i, train_data in enumerate(train_loader) :
            print("Training : {} / {}".format(i + 1, len(train_loader)), end="\r")
            train_x, _ = train_data
            # with tf.GradientTape() as tape :
            #     recon = model(train_x)
            #     loss, _, _, _ = LOSS(recon, train_x)
            #     train_loss.append(loss)

            # gradients = tape.gradient(loss, model.trainable_variables)
            # OPTIMIZER.apply_gradients(zip(gradients, model.trainable_variables))
            if model_key == 'USAD' :
                loss1, loss2 = train_step_usad(train_x, epoch)
                train_loss.append(loss1)
                train_loss2.append(loss2)
            else :
                loss = train_step(train_x)
                train_loss.append(loss)

        print("Training is completed...")
        train_loader.on_epoch_end()

        val_loss = []
        if model_key == 'USAD' :
            val_loss2 = []
        for j, val_data in enumerate(val_loader) :
            print("Validation : {} / {}".format(j + 1, len(val_loader)), end="\r")
            val_x, _ = val_data

            # recon = model(val_x)
            # loss, _, _, _ = LOSS(recon, val_x)
            if model_key == 'USAD' :
                loss1, loss2 = val_step_usad(val_x, epoch)
                val_loss.append(loss1)
                val_loss2.append(loss2)
            else :
                loss = val_step(val_x)
                val_loss.append(loss)

        val_loader.on_epoch_end()
        print("Validation is completed...")


        # save results
        train_loss_avg = sum(train_loss) / len(train_loss)
        val_loss_avg = sum(val_loss) / len(val_loss)
        if model_key == 'USAD' :
            train_loss2_avg = sum(train_loss2) / len(train_loss2)
            val_loss2_avg = sum(val_loss2) / len(val_loss2)
            results['train_loss1'].append(train_loss_avg.numpy())
            results['train_loss2'].append(train_loss2_avg.numpy())
            results['val_loss1'].append(val_loss_avg.numpy())
            results['val_loss2'].append(val_loss2_avg.numpy())

        model.save_weights(os.path.join(save_path, "{}epoch_weights".format(epoch + 1)))

        if epoch > 0:
            if model_key == 'USAD' :
                if (val_loss_avg+val_loss2_avg)/2 < min(results['val_loss']):
                    model.save_weights(os.path.join(save_path, "Best_weights".format(epoch + 1)))

            else :
                if val_loss_avg.numpy() < min(results['val_loss']):
                    # print(val_loss_avg.numpy(), min(results['val_loss']))
                    # save best weights
                    model.save_weights(os.path.join(save_path, "Best_weights".format(epoch +1)))

        if model_key == 'USAD' :
            results['train_loss'].append(((train_loss_avg+train_loss2_avg)/2).numpy())
            results['val_loss'].append(((val_loss_avg+val_loss2_avg)/2).numpy())

        else :
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
