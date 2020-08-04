from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from keras import backend as K
from keras import Sequential

import pickle
import numpy as np

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

root_dir = '/home/renyi/Documents/InceptionTime/'
latent_dim = 10
batch_size = 1
intermediate_dim = 70
T = 100

x = Input(batch_shape=(batch_size, T))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)

    return z_mean + K.exp(z_log_sigma) * epsilon

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss



z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(T, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(x, x_decoded_mean)
vae2 = Model(x, x_decoded_mean)

'''
encoder = Model(x, z_mean)
decoder_input = Input(shape=(latent_dim, ))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
'''

vae.compile(optimizer='rmsprop', loss=vae_loss)
vae2.compile(optimizer='rmsprop', loss=vae_loss)

datasets_dict = {}
z_datasets_dict = {}

for dataset_name in DATASET_NAMES: # univariate dataset names
    file_name = root_dir + 'data/' + dataset_name + '.pickle'
    with open(file_name, 'rb') as f:
        datasets_dict[dataset_name] = pickle.load(f)

def transform_data(model, data):
    intermediate1 = Model(inputs=model.input, outputs=model.layers[2].output)
    intermediate1_output = intermediate1.predict(data)
    intermediate2 = Model(inputs=model.input, outputs=model.layers[3].output)
    intermediate2_output = intermediate2.predict(data)

    result = np.concatenate((intermediate1_output, intermediate2_output), axis=1)

    return result


for key in datasets_dict:
    dataset = datasets_dict[key]
    xtrain, ytrain, xtest, ytest = dataset
    print('\nRunning ', key)
    vae.fit(xtrain, xtrain, shuffle=True, epochs=1, batch_size=1, validation_data=(xtest,xtest))
    vae2.fit(xtest, xtest,shuffle=True, epochs=1, batch_size = 1)
    
    ztrain = transform_data(vae, xtrain)
    ztest = transform_data(vae2, xtest)

    z_datasets_dict[key] = (ztrain, ytrain, ztest, ytest)

for dsname in z_datasets_dict:
    file_name = root_dir + '/data/' + dsname + '.pickle'
    # os.makedirs(output_directory)
    with open(file_name, 'wb') as f:
        pickle.dump(z_datasets_dict[dsname], f)

'''
for key in z_datasets_dict:
    ztrain, ytrain, ztest, ytest = z_datasets_dict[key]

    final_model = Sequential()
    final_model.add(Dense(30, input_dim = 20, activation='relu'))
    final_model.add(Dense(15, activation='relu'))
    final_model.add(Dense(1))

    final_model.compile(loss='mean_squared_error', optimizer='adam')
    final_model.fit(ztrain, ytrain)
    prediction = final_model.predict(ztest)
    print(prediction - ytest)
'''