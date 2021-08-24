import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam

def rmse_loss(recon, origin) :
    n_timewindow = origin.shape[1]
    n_feature = origin.shape[2]

    # calculate rmse
    error = tf.math.subtract(recon, origin)
    error = tf.math.pow(error, 2)
    error = tf.math.reduce_sum(error, axis = 1)
    error = tf.math.reduce_sum(error, axis = 1)
    error = tf.math.divide(error, (n_timewindow*n_feature))
    error = tf.math.sqrt(error)

    # calculate mean of rmse value by batch for train
    error_mean = tf.reduce_mean(error)

    # calculate median and maximum of rmse value by batch for test
    error_array = np.array(error)
    error_median = np.median(error_array)
    error_maximum = np.max(error_array)
    error_minimum = np.min(error_array)

    return error_mean, error_median, error_maximum, error_minimum

def get_metric(key) :
    if key == 'rmse' :
        LOSS = rmse_loss

    return LOSS

def get_optimizer(key, lr) :
    if key == 'adam' :
        OPTIMIZER = Adam(learning_rate=lr)

    return OPTIMIZER


class Dataloader(Sequence) :
    def __init__(self, path, label = False, timewindow = 100, batch_size=1, shuffle=True):
        self.path = path
        self.label = label
        self.data_list = glob.glob(os.path.join(self.path, '*.csv'))
        self.timewindow = timewindow
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data_list)) / float(self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))

        if self.shuffle :
            np.random.shuffle(self.indices)

    def get_inputs(self, path):
        filename = os.path.basename(path)
        df = pd.read_csv(path, index_col=0)
        col_list = ['HEV_AccelPdlVal', 'HEV_EngSpdVal', \
                   'IEB_StrkDpthPcVal',
                   'WHL_SpdFLVal', \
                   'NTU', 'NAB', 'ntug_SyncFilt']

        if not list(df.columns) == col_list :
            df.columns = col_list

        # delete brake pedal signal
        col_list_rev = ['HEV_AccelPdlVal', 'HEV_EngSpdVal', \
                    # 'IEB_StrkDpthPcVal',
                    'WHL_SpdFLVal', \
                    'NTU', 'NAB', 'ntug_SyncFilt']

        df = df.loc[:, col_list_rev]

        array = np.array(df)

        inputs = []

        n_samples = array.shape[0] - self.timewindow + 1

        for n in range(n_samples) :
            inputs.append(array[n:n+self.timewindow])
        inputs = np.array(inputs)

        return inputs, filename


    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        i = indices[0]
        # for i in indices :
        batch_x, filename = self.get_inputs(self.data_list[i])

        if self.label :
            if os.path.basename(self.data_list[i]).split('_')[0] == 'abnormal' :
                batch_y = True
            else:
                batch_y = False
            return batch_x, batch_y, filename
        else :
            return batch_x, batch_x



def gpu_limit(GB) :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")
    # set the only one GPU and memory limit
    memory_limit = 1024 * GB
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU is not available')

def dir_exist_check(paths) :
    for path in paths :
        if not os.path.isdir(path) :
            os.makedirs(path)

class WriteResults() :
    def __init__(self, test_results, results_dic, save_path):
        self.df_total = pd.DataFrame(test_results)
        self.dic = results_dic
        self.standards = self.dic.keys()
        self.save_path = save_path
        self.draw_pdf()
        self.write_rep()


    def draw_pdf(self):
        df_normal = self.df_total[(self.df_total['label'] == False)].copy()
        df_abnormal = self.df_total[(self.df_total['label'] == True)].copy()
        for standard in self.standards :

            pyplot.figure()
            sns.distplot(df_normal[standard], label='normal', color='green')
            sns.distplot(df_abnormal[standard], label='abnormal', color='red')

            pyplot.legend(prop={'size': 14})
            pyplot.savefig(os.path.join(self.save_path, standard+'.png'))
            # pyplot.show()

    def write_rep(self):
        self.rep = {}
        self.rep['standard'] = []
        self.rep['F1'] = []
        self.rep['Precision'] = []
        self.rep['Recall'] = []
        self.rep['AUROC'] = []

        pyplot.figure()
        recalls = []
        b_fprs = []

        for standard in self.standards :
            b_f1 = self.dic[standard]['F1'].max()
            idx = list(self.dic[standard]['F1']).index(b_f1)
            precision = self.dic[standard].iloc[idx]['Precision']
            recall = self.dic[standard].iloc[idx]['Recall']
            b_fpr = self.dic[standard].iloc[idx]['FRR']

            recalls.append(recall)
            b_fprs.append(b_fpr)

            TPR = list(self.dic[standard]['Recall'])
            FPR = list(self.dic[standard]['FRR'])

            # get AUROC score
            h_flag = False
            x_st = FPR[0]
            auc = 0
            for i in range(len(TPR)-1) :
                x = FPR[i+1]
                x_prev = FPR[i]

                y = TPR[i+1]
                y_prev = TPR[i]

                if y > y_prev :
                    if h_flag :
                        x_d = x_prev - x_st
                        auc += x_d * y_prev
                    x_st = x

                else :
                    h_flag = True

                    if i == len(TPR)-2 :
                        x_d = x - x_st
                        auc += x_d * y
            ####
            self.rep['standard'].append(standard)
            self.rep['F1'].append(b_f1)
            self.rep['Precision'].append(precision)
            self.rep['Recall'].append(recall)
            self.rep['AUROC'].append(auc)

            # pyplot.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', label='No Skill')
            pyplot.plot(FPR, TPR, marker='.', label=standard)

        pyplot.scatter(b_fprs, recalls, c='k', zorder=9, label = 'set to maximize the F1 Score')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        pyplot.savefig(os.path.join(self.save_path, 'ROC.png'))

        df = pd.DataFrame(self.rep)
        df.set_index('standard', inplace=True)
        df.to_csv(os.path.join(self.save_path, 'test_results.csv'))


class results_analysis() :
    def __init__(self, results):
        self.df_total = pd.DataFrame(results)
        # print(self.df_total)
        self.df_normal = self.df_total[(self.df_total['label'] == False)]
        self.df_abnormal = self.df_total[(self.df_total['label'] == True)]
        self.standards = ['mean', 'median', 'maximum', 'minimum']
        self.confusion_m_type =['CD', 'MD', 'FA', 'CR', 'Accuracy', 'Precision', 'Recall', 'F1', 'FRR', 'FAR']
        self.confusion_m_init()
        self.results_init()

    def results_init(self):
        self.results = {}

    def confusion_m_init(self):
        self.confusion_m = {}
        for confusion in self.confusion_m_type :
            self.confusion_m[confusion] = []

    def get_metric(self):
        for standard in self.standards :
            df_total_sorted = self.df_total.copy()
            df_total_sorted.sort_values(by=standard, ascending=False, inplace=True)
            df_total_sorted.reset_index(drop=True, inplace=True)

            for i in range(len(self.df_total)) :
                # Alarm lists
                df_A = df_total_sorted.copy()[:i+1]
                CD = len(df_A[df_A['label'] == True])
                FA = len(df_A[df_A['label'] == False])
                self.confusion_m['CD'].append(CD)
                self.confusion_m['FA'].append(FA)

                # Reject lists
                df_R = df_total_sorted.copy()[i+1:]
                MD = len(df_R[df_R['label'] == True])
                CR = len(df_R[df_R['label'] == False])
                self.confusion_m['MD'].append(MD)
                self.confusion_m['CR'].append(CR)

                # metric lists
                self.confusion_m['Accuracy'].append((CD + CR) / (MD + FA))
                Precision = CD / (CD + FA)
                Recall = CD / (CD + MD)

                if Precision == 0.0 and Recall == 0.0 :
                    F1 = 0.0
                else :
                    F1 = 2 * (Recall * Precision) / (Recall + Precision)

                self.confusion_m['Precision'].append(Precision)
                self.confusion_m['Recall'].append(Recall)
                self.confusion_m['F1'].append(F1)
                self.confusion_m['FRR'].append(FA / (FA + CR))
                self.confusion_m['FAR'].append(MD / (CD + MD))


            confusion_df = pd.DataFrame(self.confusion_m)
            self.results[standard] = confusion_df
            self.confusion_m_init()

        return self.results