import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from detecta import detect_cusum
import inspect

root_dir = '/home/renyi/Documents/InceptionTime/'

###### 
thetas = [1, 0.75, 0.5, 0.25, -1]
L1_errors = []

count = 0
for dataset_name in DATASET_NAMES: # univariate dataset names
    theta = dataset_name.split('_')[1][6:]
    count += 1

    print('Showing result of dataset ', dataset_name)
    pred_file = root_dir + 'results/inception/TSC/' + dataset_name + '/y_pred.npy'
    data_file = root_dir + 'data/' + dataset_name + '.pickle'
    
    raw_predictions = np.load(pred_file)

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        data_to_plot = data[0][500]
        
        '''
        plt.scatter(range(len(data_to_plot)), data_to_plot)
        plt.axvline(data[3][500], label='eta_i = ' + str(data[3][500]))
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('value')
        plt.title('sample chosen randomly from ' + dataset_name)

        plt.show()
        plt.close()
        '''
        

        n, T = data[2].shape
        ys = data[3] # classes in original change points

    nclasses = raw_predictions.shape[1]

    if nclasses == 1: # regression
        ABS = 1/n * np.sum(np.abs(ys - (raw_predictions.T)[0]))
        L1_errors.append(ABS)

        print('ABS: ', ABS)
    else: # classification
        # predictions of classes in 0, 1, 2, etc
        class_predictions = (np.argmax(raw_predictions, axis=1) + 1)
        diff = class_predictions - ys
        ABS = 1/n * np.sum(np.abs(diff))
        L1_errors.append(ABS)
        
        print('Error rate: ', np.count_nonzero(diff) / n)
        print('ABS: ', ABS)
    
    if (count == 5):
        plt.scatter(thetas, L1_errors)
        plt.xlabel('theta')
        plt.ylabel('L1 error')
        plt.show()
        plt.close()
        count = 0
        L1_errors = []


    '''
    lines = inspect.getsource(detect_cusum)

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        ys = data[3]

        drifts = [0, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03]

        for drift in drifts:
            print('Running drift ', drift)

            count_1 = 0 # number of predictions that have 1 change point
            error = 0 # total error
            for i in range(len(ys)):
                x = data[2][i]
                ta, tai, taf, amp = detect_cusum(x, 1, drift, False, False)
                if len(ta) == 1:
                    count_1 += 1
                    error += abs(ta[0] - ys[i])
            print('\tPercent of samples that are predicted with 1 change point: ')
            print(count_1 / len(ys))
            print('\tError: ')
            print(error / count_1)
            print()
    '''