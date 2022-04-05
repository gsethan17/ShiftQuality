import os
from model import get_model
from utils import get_metric, Dataloader


if __name__ == '__main__' : 
    ##### neen to set the benchmark number ########
    # 0 or 1
    # 0 : 'F1', 1 : 'AUROC'
    benchmark_num = 0
    ##### neen to set the benchmark number ########

    benchmark_list = ['F1', 'AUROC']
    benchmark_key = benchmark_list[benchmark_num]

    model_key = 'USAD'
    
    print(model_key, benchmark_key)

    if benchmark_key == 'F1' :
        n_timewindow = 50
        epoch = 40

        standard = 'median'

        alpha = 0.9
    
    elif benchmark_key == 'AUROC' :
        n_timewindow = 70
        epoch = 20

        standard = 'minimum'

        alpha = 0.0


    n_feature = 6
    latent_size = 10
    

    weight_path = os.path.join(os.getcwd(), 'results', model_key, str(n_timewindow), '0.001_' + str(epoch), 'train', 'Best_weights')

    model = get_model(model_key, n_timewindow, n_feature, latent_size)
    print('----------- model is loaded ---------')


    model.load_weights(weight_path)
    print('---------- model weight is set ---------')


    LOSS = get_metric('rmse', model_key, ['USAD'])

    
    ######### input data is needed ############
    # input shape : (None, n_timewindow, n_feature)
    #X = ##input##
    
    ### sample ###
    test_data_path = '/home/gsethan/Documents/ShiftQuality/test/'+str(n_timewindow)
    test_loader = Dataloader(test_data_path, label=True, timewindow = n_timewindow)

    X, Y, filename = test_loader[0]
    ### sample ###
    ######### input data is needed ############



    ## inference code
    w1, w2, w3 = model(X)

    ## refer to page 22 of my thesis^^
    ## Algorithm 2 is interence pseudo code
    ## w1 = AE_{1}(X_{test})
    ## w2 = AE_{2}(X_{test})
    ## w3 = AE_{2}(AE_{1}(X_{test}))

    ## score = ((1-alpha)*L(X,w1)) + (alpha*L(X,w3))

    mean, median, maximum, minimum = LOSS(step=3, recon=w1, rerecon=w3, origin=X, a=alpha)

    if standard == 'median' :
        score = median

    elif standard == 'minimum' :
        score = minimum

    print(score, filename, Y)

