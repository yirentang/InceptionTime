import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

root_dir = '/home/renyi/Documents/InceptionTime/'

###### 
beta = [1, 0.75, 0.5, 0.25]
L1_error = []


for dataset_name in DATASET_NAMES: # univariate dataset names
    print('Showing result of dataset ', dataset_name)
    pred_file = root_dir + 'results/inception/TSC/' + dataset_name + '/y_pred.npy'
    data_file = root_dir + 'data/' + dataset_name + '.pickle'
    
    raw_predictions = np.load(pred_file)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        data_to_plot = data[2][500]
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
        if dataset_name != 'SIMULATED_mdiff=-1':
            L1_error.append(ABS)

        print('ABS: ', ABS)
    else: # classification
        # predictions of classes in 0, 1, 2, etc
        class_predictions = (np.argmax(raw_predictions, axis=1) + 1)
        diff = class_predictions - ys
        ABS = 1/n * np.sum(np.abs(diff))
        if dataset_name != 'SIMULATED_mdiff=-1':
            L1_error.append(ABS)
        
        print('Error rate: ', np.count_nonzero(diff) / n)
        print('ABS: ', ABS)


plt.scatter(beta, L1_error)
plt.xlabel('theta')
plt.ylabel('L1 error')
plt.title('theta vs L1 error')
plt.show()
plt.close()