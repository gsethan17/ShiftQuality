from utils import Dataloader, gpu_limit, dir_exist_check, get_metric, get_optimizer, rmse_loss
from model import get_model, get_lstm_model, LSTM_generator, LSTM_discriminator, get_shallow_model
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


def train(model_key, i, j) :
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
    # n_timewindow = int(config['DATA']['N_TIMEWINDOW'])
    n_timewindow = j
    n_feature = int(config['DATA']['N_FEATURE'])
    latent_size = int(config['DATA']['LATENT_SIZE'])

    train_path = os.path.join(train_path, str(n_timewindow))
    val_path = os.path.join(val_path, str(n_timewindow))

    ## model setup
    # model_key = config['MODEL']['KEY']
    USADs = ['USAD', 'USAD-LSTM']


    ## train setup
    metric = config['TRAIN']['METRIC']
    LOSS = get_metric(metric, model_key, USADs)

    optimizer = config['TRAIN']['OPTIMIZER']
    learning_rate = float(config['TRAIN']['LEARNING_RATE'])
    OPTIMIZER = get_optimizer(optimizer, learning_rate)

    # epochs = int(config['TRAIN']['EPOCHS'])
    epochs = i

    # save path setup
    save_path_base = os.path.join(os.getcwd(), 'full_data', model_key, str(n_timewindow), str(learning_rate) + '_' + str(epochs))
    save_path = os.path.join(os.getcwd(), 'full_data', model_key, str(n_timewindow), str(learning_rate) + '_' + str(epochs), 'train')
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
    '''
    if model_key == 'LSTM-AE' :
        model = get_lstm_model(n_timewindow, n_feature, latent_size)
    else :
        model = get_model(model_key, n_timewindow, n_feature, latent_size)
    '''
    model = get_shallow_model(model_key, n_timewindow, n_feature, latent_size)
    print("Model is loaded!")

    results = {}
    results['train_loss'] = []
    results['val_loss'] = []
    if model_key in USADs :
        results['train_loss1'] = []
        results['train_loss2'] = []
        results['val_loss1'] = []
        results['val_loss2'] = []

    for epoch in range(epochs) :
        train_loss = []
        if model_key in USADs :
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
            if model_key in USADs :
                loss1, loss2 = train_step_usad(train_x, epoch)
                train_loss.append(loss1)
                train_loss2.append(loss2)
            else :
                loss = train_step(train_x)
                train_loss.append(loss)

        print("Training is completed...")
        train_loader.on_epoch_end()

        val_loss = []
        if model_key in USADs :
            val_loss2 = []
        for j, val_data in enumerate(val_loader) :
            print("Validation : {} / {}".format(j + 1, len(val_loader)), end="\r")
            val_x, _ = val_data

            # recon = model(val_x)
            # loss, _, _, _ = LOSS(recon, val_x)
            if model_key in USADs :
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
        if model_key in USADs :
            train_loss2_avg = sum(train_loss2) / len(train_loss2)
            val_loss2_avg = sum(val_loss2) / len(val_loss2)
            results['train_loss1'].append(train_loss_avg.numpy())
            results['train_loss2'].append(train_loss2_avg.numpy())
            results['val_loss1'].append(val_loss_avg.numpy())
            results['val_loss2'].append(val_loss2_avg.numpy())

        model.save_weights(os.path.join(save_path, "{}epoch_weights".format(epoch + 1)))

        if epoch > 0:
            if model_key in USADs :
                if (val_loss_avg+val_loss2_avg)/2 < min(results['val_loss']):
                    model.save_weights(os.path.join(save_path, "Best_weights".format(epoch + 1)))

            else :
                if val_loss_avg.numpy() < min(results['val_loss']):
                    # print(val_loss_avg.numpy(), min(results['val_loss']))
                    # save best weights
                    model.save_weights(os.path.join(save_path, "Best_weights".format(epoch +1)))

        if model_key in USADs :
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
        if not model_key in USADs :
            if epoch > 5:
                if results['val_loss'][-5] < min(results['val_loss'][-4:]):
                    print(results['val_loss'][-5])
                    print(min(results['val_loss'][-4:]))
                    break

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)

def train_MAD_GAN() :
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
    test_data_path = config['PATH']['TEST']
    test_data_path = os.path.join(test_data_path, str(n_timewindow))

    ## model setup
    model_key = config['MODEL']['KEY']

    ## train setup
    metric = config['TRAIN']['METRIC']
    LOSS = tf.keras.losses.BinaryCrossentropy()

    optimizer = config['TRAIN']['OPTIMIZER']
    learning_rate = float(config['TRAIN']['LEARNING_RATE'])
    OPTIMIZER = get_optimizer(optimizer, learning_rate)

    epochs = int(config['TRAIN']['EPOCHS'])

    # save path setup
    save_path_base = os.path.join(os.getcwd(), 'results', 'MAD_GAN', model_key, str(n_timewindow),
                                  str(learning_rate))
    save_path = os.path.join(os.getcwd(), 'results', 'MAD_GAN', model_key, str(n_timewindow),
                             str(learning_rate), 'train')
    save_path_test = os.path.join(save_path_base, 'test')
    dir_exist_check([save_path, save_path_test])

    # GPU limitation
    limit_gb = int(config['GPU']['LIMIT'])
    # gpu_limit(limit_gb)

    # save parameters
    param = {}
    param['special'] = 'MAD-GAN'
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
    test_loader = Dataloader(test_data_path, label=True, timewindow=n_timewindow)

    print(len(train_loader), len(val_loader))

    # MODEL LOADER
    generator = LSTM_generator(model_key, n_timewindow, n_feature)
    discriminator = LSTM_discriminator(model_key, n_timewindow, n_feature)

    generator.compile(loss=LOSS, optimizer=OPTIMIZER)
    discriminator.compile(loss=LOSS, optimizer=OPTIMIZER)

    ### dictionary

    # train
    for epoch in range(epochs) :

        for i, train_data in enumerate(train_loader) :
            print("Training : {} / {}".format(i + 1, len(train_loader)), end="\r")
            train_x, _ = train_data

            batch_size = train_x.shape[0]
            z = tf.random.normal(shape=[batch_size, n_timewindow, latent_size])

            g_z = generator(z)
            g_z_X = tf.concat([g_z, train_x], axis = 0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(g_z_X, y1)

            z = tf.random.normal(shape=[batch_size, n_timewindow, latent_size])
            y2 = tf.constant([[1.]] * batch_size)
            generator.train_on_batch(z, y2)


        train_loader.on_epoch_end()

        if epoch % 10 == 9 :
            # save weights
            temp_path = os.path.join(save_path_test, '{}'.format(epoch+1))
            generator.save_weights(os.path.join(temp_path, 'generator'))
            discriminator.save_weights(os.path.join(temp_path, 'discriminator'))

            # test_MAD_GAN(generator, discriminator, test_loader, n_timewindow, latent_size)
    print("Training is completed...")

def test_MAD_GAN(generator, discriminator, test_loader, n_timewindow, latent_size) :

    recon_loss = rmse_loss

    for i, test_data in enumerate(test_loader):
        print("Testing : {} / {}".format(i + 1, len(test_loader)), end="\r")
        test_x, test_y, filename = test_data

        # mapping testing data to latent space
        batch_size = test_x.shape[0]

        losses = []
        for i in range(1000) :
            val_normal_list = random.sample(normal_data_list, int(len(normal_data_list)*0.2))
            z = tf.random.normal(shape=[batch_size, n_timewindow, latent_size])
            g_z = generator(z)

            loss = tf.keras.losses.MeanSquaredError(test_x, g_z)
            losses.append(loss)

            if len(losses) > 0 :
                if loss == min(losses) :
                    g_z_opt = g_z
                    loss_opt = loss

        d_x = discriminator(test_x)
        y = tf.constant([[0.]] * batch_size)
        tf.keras.losses.BinaryCrossentropy(y, d_x)




if __name__ == '__main__' :
    # model_keys = ['RNN-AE', 'LSTM-AE', 'GRU-AE', 'AE']
    model_keys = ['AE']
    for model_key in model_keys :
        for i in range(10, 110, 10) :
            train(model_key, 50, i)
    '''
    # epoch 10 to 50
    for i in range(10, 60, 10) :

        # timewindow 10 to 100
        for j in range(10, 110, 10) :
            print(i, j)
            train(i, j)
    # train_MAD_GAN()
    '''
