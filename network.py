import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, ToTensor

import pickle
from itertools import cycle
import random

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

class Encoder(nn.Module):
    def __init__(self, mdim, cdim, sdim):
        super(Encoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(784, mdim),
            nn.Tanh()
        )
        self.content_mu = nn.Linear(mdim, cdim)
        self.content_logvar = nn.Linear(mdim, cdim)

        self.style_mu = nn.Linear(mdim, sdim)
        self.style_logvar = nn.Linear(mdim, sdim)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        z_content_mu = self.content_mu(x)
        z_content_logvar = self.content_logvar(x)
        z_style_mu = self.style_mu(x)
        z_style_logvar = self.style_logvar(x)

        return z_content_mu, z_content_logvar, z_style_mu, z_style_logvar

class Decoder(nn.Module):
    def __init__(self, mdim, cdim, sdim):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(cdim + sdim, mdim),
            nn.Tanh(),
            nn.Linear(mdim, 784),
            nn.Sigmoid()
        )
    
    def forward(self, z_content, z_style):
        x = torch.cat((z_content, z_style), dim=1)
        x = self.fc(x)
        #x = x.view(x.size(0))

        return x

class DoubleUniNormal(Dataset):
    def __init__(self, dsname):
        file_name = root_dir + 'data/original/' + dsname + '.pickle'
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
        self.x_train, self.y_train, self.x_test, self.y_test = dataset
    
    def __len__(self):
        return self.x_train.size

    def __getitem__(self, idx):
        _, T = self.x_train.shape
        row = idx // T
        column = idx % T
        if column < self.y_train[row]:
            label = 2*row
        else:
            label = 2*row + 1
        return (self.x_train[row][column], label)

class experiment1(Dataset):
    def __init__(self):
        self.mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
        self.mnist.data = np.true_divide(self.mnist.data, 255)
        # data is X, label is y
        outputs_to_concat = []
        for idx in range(5):
            indices1 = self.mnist.targets == 2*idx
            tmp1 = self.mnist.data[indices1]
            first_5000 = tmp1.view(tmp1.size(0), -1)[0:5000]
            first_5000 = torch.transpose(first_5000, 0, 1)

            indices2 = self.mnist.targets == (2*idx+1)
            tmp2 = self.mnist.data[indices2]
            second_5000 = tmp2.view(tmp2.size(0), -1)[0:5000]
            second_5000 = torch.transpose(second_5000, 0, 1)
        
            row = torch.cat((first_5000, second_5000), dim=1)
            outputs_to_concat.append(row)

        self.sample = torch.stack(outputs_to_concat, dim=0)

    def __len__(self):
        return 50000
    
    def __getitem__(self, idx):
        d1 = idx // 10000
        d2 = idx % 10000
        if d2 < 5000:
            label = 2*d1
        else:
            label = 2*d1 + 1

        # print(self.sample[d1, :, d2])
        return (self.sample[d1, :, d2], label)

class experiment2(Dataset):
    def __init__(self, n):
        self.n = n
        self.mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
        self.mnist.data = np.true_divide(self.mnist.data, 255)
        self.labels = []

        # data is X, label is y
        outputs_to_concat = []
        for i in range(n):
            candidates = random.sample(range(10), 2)
            i1 = min(candidates)
            i2 = max(candidates)

            indices1 = self.mnist.targets == i1
            tmp1 = self.mnist.data[indices1]
            first_5000 = tmp1.view(tmp1.size(0), -1)[0:5000]
            first_5000 = torch.transpose(first_5000, 0, 1)

            indices2 = self.mnist.targets == i2
            tmp2 = self.mnist.data[indices2]
            second_5000 = tmp2.view(tmp2.size(0), -1)[0:5000]
            second_5000 = torch.transpose(second_5000, 0, 1)
        
            row = torch.cat((first_5000, second_5000), dim=1)
            outputs_to_concat.append(row)

            self.labels.append([i1, i2])

        self.sample = torch.stack(outputs_to_concat, dim=0)
        print('wd')

    def __len__(self):
        return 10000*self.n
    
    def __getitem__(self, idx):
        d1 = idx // 10000
        d2 = idx % 10000
        if d2 < 5000:
            label = self.labels[d1][0]
        else:
            label = self.labels[d1][1]

        # print(self.sample[d1, :, d2])
        return (self.sample[d1, :, d2], label)



def accumulate_group_evidence(class_mu, class_logvar, labels_batch):
    """
    :param class_mu: mu values for class latent embeddings of each sample in the mini-batch
    :param class_logvar: logvar values for class latent embeddings for each sample in the mini-batch
    :param labels_batch: class labels of each sample (the operation of accumulating class evidence can also
        be performed using group labels instead of actual class labels)
    :return:
    """
    var_dict = {}
    mu_dict = {}

    # convert logvar to variance for calculations
    class_var = class_logvar.exp_()

    # calculate var inverse for each group using group vars
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()
        # remove 0 values from variances
        class_var[i][class_var[i] == float(0)] = 1e-6
        if group_label in var_dict.keys():
            var_dict[group_label] += 1 / class_var[i]
        else:
            var_dict[group_label] = 1 / class_var[i]

    # invert var inverses to calculate mu and return value
    for group_label in var_dict.keys():
        var_dict[group_label] = 1 / var_dict[group_label]

    # calculate mu for each group
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()
        if group_label in mu_dict.keys():
            mu_dict[group_label] += class_mu[i] * (1 / class_var[i])
        else:
            mu_dict[group_label] = class_mu[i] * (1 / class_var[i])

    # multiply group var with sums calculated above to get mu for the group
    for group_label in mu_dict.keys():
        mu_dict[group_label] *= var_dict[group_label]

    # replace individual mu and logvar values for each sample with group mu and logvar
    group_mu = torch.FloatTensor(class_mu.size(0), class_mu.size(1))
    group_var = torch.FloatTensor(class_var.size(0), class_var.size(1))

    
    group_mu = group_mu.cuda()
    group_var = group_var.cuda()

    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        group_mu[i] = mu_dict[group_label]
        group_var[i] = var_dict[group_label]

        # remove 0 from var before taking log
        group_var[i][group_var[i] == float(0)] = 1e-6

    # convert group vars into logvars before returning
    return Variable(group_mu, requires_grad=True), Variable(torch.log(group_var), requires_grad=True)

def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def group_wise_reparameterize(training, mu, logvar, labels_batch, cuda):
    eps_dict = {}

    # generate only 1 eps value per group label
    for label in torch.unique(labels_batch):
        if cuda:
            eps_dict[label.item()] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)
        else:
            eps_dict[label.item()] = torch.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)

    if training:
        std = logvar.mul(0.5).exp_()
        reparameterized_var = Variable(std.data.new(std.size()))

        # multiply std by correct eps and add mu
        for i in range(logvar.size(0)):
            reparameterized_var[i] = std[i].mul(Variable(eps_dict[labels_batch[i].item()]))
            reparameterized_var[i].add_(mu[i])

        return reparameterized_var
    else:
        return mu

def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()

def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement() # this should be the batch size

def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.1)
        layer.bias.data.zero_()

def loss_function(content_mu, content_logvar, style_mu, style_logvar, group_mu, group_logvar):
    style_KL = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
    style_KL /= bsize
    content_KL = -0.5 * torch.sum(1 + group_logvar - group_mu.pow(2) - group_logvar.exp())
    content_KL /= bsize


transform_config = Compose([ToTensor()])
root_dir = '/home/renyi/Documents/InceptionTime/'
bsize = 256


dataset = experiment2(50)
loader = cycle(DataLoader(dataset, batch_size=bsize, shuffle=True, drop_last=True))

# build network model
encoder = Encoder(400, 10, 10)
encoder.apply(weights_init)
decoder = Decoder(400, 10, 10)
decoder.apply(weights_init)
X = torch.FloatTensor(bsize, 784)

# move to GPU
encoder.cuda()
decoder.cuda()
X = X.cuda()

# define optimizer
encoder_optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
)

losses = []
# training
for epoch in range(50):
    print('\nEpoch', epoch)
    train_loss = 0
    for iteration in range(int(len(dataset) / bsize)):
        xs_batch, labels_batch = next(loader)
        X.copy_(xs_batch)
        # xs = xs_batch.view(xs_batch.size(0), 1)
        # X.copy_(xs)

        encoder_optimizer.zero_grad()
        
        content_mu, content_logvar, style_mu, style_logvar = encoder(Variable(X))
        group_mu, group_logvar = accumulate_group_evidence(content_mu.data,
        content_logvar.data, labels_batch)

        style_KL = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
        style_KL /= bsize

        content_KL = -0.5 * torch.sum(1 + group_logvar - group_mu.pow(2) - group_logvar.exp())
        content_KL /= bsize

        style_latent = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
        content_latent = group_wise_reparameterize(
            training=True, mu=group_mu, logvar=group_logvar, labels_batch=labels_batch, cuda=True
        )
        reconstructed_xs = decoder(style_latent, content_latent)
        temp = Variable(X).view(X.size(0), -1)
        reconstruction_error = mse_loss(reconstructed_xs, temp)
        
        total_loss = style_KL + content_KL + reconstruction_error
        total_loss.backward()

        train_loss += total_loss.item()

        encoder_optimizer.step()

        if iteration + 1 == int(len(dataset) / bsize):
            # print(reconstructed_xs.view(1, reconstructed_xs.size(0)))
            # print(X.view(1, X.size(0)))
            '''
            for vector in reconstructed_xs - X:
                print(vector)
            '''
            print(train_loss)
            print('Iteration', iteration)
            print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
            print('Style KL-Divergence loss: ' + str(style_KL.data.storage().tolist()[0]))
            print('Class KL-Divergence loss: ' + str(content_KL.data.storage().tolist()[0]))
    losses.append(train_loss)

x = range(50)

plt.scatter(x, losses, s=0.8)
plt.show()


'''
for dsname in DATASET_NAMES:
    print('Loading ', dsname)
    dataset = DoubleUniNormal(dsname)
    loader = cycle(DataLoader(dataset, batch_size=bsize, shuffle=True, drop_last=True))

    # build network model
    encoder = Encoder(4, 1, 1)
    encoder.apply(weights_init)
    decoder = Decoder(4, 1, 1)
    decoder.apply(weights_init)
    X = torch.FloatTensor(bsize, 1)
    
    # move to GPU
    encoder.cuda()
    decoder.cuda()
    X = X.cuda()

    # define optimizer
    encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
    )

    losses = []
    # training
    for epoch in range(100):
        print('\nEpoch', epoch)
        train_loss = 0
        for iteration in range(int(len(dataset) / bsize)):
            xs_batch, labels_batch = next(loader)
            xs = xs_batch.view(xs_batch.size(0), 1)
            # xs_4fold = torch.cat((xs, xs, xs, xs), 1)
            X.copy_(xs)

            encoder_optimizer.zero_grad()
            
            content_mu, content_logvar, style_mu, style_logvar = encoder(Variable(X))
            group_mu, group_logvar = accumulate_group_evidence(content_mu.data,
            content_logvar.data, labels_batch)

            style_KL = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
            style_KL /= bsize

            content_KL = -0.5 * torch.sum(1 + group_logvar - group_mu.pow(2) - group_logvar.exp())
            content_KL /= bsize

            style_latent = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
            content_latent = group_wise_reparameterize(
                training=True, mu=group_mu, logvar=group_logvar, labels_batch=labels_batch, cuda=True
            )
            reconstructed_xs = decoder(style_latent, content_latent)
            temp = Variable(X).view(X.size(0), -1)
            reconstruction_error = mse_loss(reconstructed_xs, temp)
            
            total_loss = style_KL + content_KL + reconstruction_error
            total_loss.backward()

            train_loss += total_loss.item()

            encoder_optimizer.step()

            if iteration + 1 == int(len(dataset) / bsize):
                print(reconstructed_xs.view(1, reconstructed_xs.size(0)))
                print(X.view(1, X.size(0)))
                #print(reconstructed_xs)
                #print(X)
                print(train_loss)
                print('Iteration', iteration)
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('Style KL-Divergence loss: ' + str(style_KL.data.storage().tolist()[0]))
                print('Class KL-Divergence loss: ' + str(content_KL.data.storage().tolist()[0]))
        losses.append(total_loss)

    x = range(100)

    plt.scatter(x, losses, s=0.8)
    plt.show()

    break
'''