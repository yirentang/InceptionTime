import numpy as np
import os
import pickle

root_dir = '/home/renyi/Documents/InceptionTime/'

# fixed mean and variance
def simulate_data(T, nclasses, mean, var):
    interval = round(T/nclasses) - round(T/nclasses**2)
    labels = interval * (np.array(range(nclasses)) + 1)
    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1500):
        label = np.random.choice(labels, 1)[0]
        generation1 = np.concatenate((np.random.normal(mean, var, label), np.random.normal(2*mean, var, T-label)))
        generation2 = np.concatenate((np.random.normal(mean, var, label), np.random.normal(2*mean, var, T-label)))
        x_train.append(generation1)
        y_train.append(label)
        x_test.append(generation2)
        y_test.append(label)
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    datasets_dict = {}
    datasets_dict['SIMULATED'] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    
    return datasets_dict

# fixed variance, mean generated differently
def generate2(T, nclasses, var, mdiff_random):
    interval = round(T/nclasses) - round(T/nclasses**2)
    labels = interval * (np.array(range(nclasses)) + 1)
    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1500):
        label = np.random.choice(labels, 1)[0]
        mean1 = np.random.uniform(0, 1)
        if mdiff_random == -1:
            mean_diff = np.random.uniform(0,1)
        else:
            mean_diff = mdiff_random
        mean2 = mean1 + mean_diff * np.random.choice([-1,1], 1)[0]
        
        generation1 = np.concatenate((np.random.normal(mean1, var, label), np.random.normal(mean2, var, T-label)))
        generation2 = np.concatenate((np.random.normal(mean1, var, label), np.random.normal(mean2, var, T-label)))
        x_train.append(generation1)
        y_train.append(label)
        x_test.append(generation2)
        y_test.append(label)
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    
    return (x_train, y_train, x_test, y_test)

def generate2wrapper(T, nclasses, var):
    datasets_dict = {}
    candidates = [-1, 0, 0.25, 0.5, 0.75, 1]
    for mdiff_random in candidates:
        datasets_dict['SIMULATED_'+'mdiff='+str(mdiff_random)] = generate2(T, nclasses, var, mdiff_random)
    
    return datasets_dict

if __name__ == "__main__":
    datasets_dict = generate2wrapper(100, 100, 0.2)
    for dsname in datasets_dict:
        file_name = root_dir + '/data/' + dsname + '.pickle'
        # os.makedirs(output_directory)
        with open(file_name, 'wb') as f:
            pickle.dump(datasets_dict[dsname], f)
