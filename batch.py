import sys
import os
import time

import numpy as np
import matplotlib.pyplot as pyplot
import math
import random
from bisect import bisect_right
import glob
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import Sampler

#from pylearn2.datasets.zca_dataset import ZCA_Dataset
#from pylearn2.utils import serial

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
        import cPickle as pickle

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve
        import pickle

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)

    # We'll now download the MNIST dataset if it is not yet available.
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)

    # We'll then load and unpickle the file.
    import gzip
    with gzip.open(filename, 'rb') as f:
        data = pickle_load(f, encoding='latin-1')

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)

    X_train = X_train.reshape((-1, 1, 28, 28))  # 50k
    X_val = X_val.reshape((-1, 1, 28, 28))      # 10k
    X_test = X_test.reshape((-1, 1, 28, 28))    # 10k

    # The targets are int64, we cast them to int8 for GPU compatibility.
    # y_train = y_train.astype(np.uint8)
    # y_val = y_val.astype(np.uint8)
    # y_test = y_test.astype(np.uint8)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        alpha = .15
        epsilon = 1e-4
        nfilters = 32
        fullnetsize = 256

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, nfilters, 5),
            nn.BatchNorm2d(nfilters, eps = epsilon, momentum = alpha),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(nfilters, nfilters, 5),
            nn.BatchNorm2d(nfilters, eps = epsilon, momentum = alpha),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p = .5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, fullnetsize),
            nn.ReLU(),
            nn.Dropout(p = .0),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fullnetsize, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def regular_data_loading(bs = 64, shuffle_train = True, shuffle_test = False, device = 'cpu'):
    # Define transformations for the training set, which includes normalization
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the full training set
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Define the size of the validation set
    validation_size = 10000
    train_size = len(full_train_dataset) - validation_size

    # Split the dataset into training and validation sets
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size])

    # Create DataLoaders for each set
    train_loader = DataLoader(dataset = train_dataset, batch_size = bs, shuffle = shuffle_train)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size = bs, shuffle = shuffle_test)

    # Load the test set
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size = bs, shuffle = shuffle_test)

    return train_dataset, validation_dataset, test_dataset, train_loader, validation_loader, test_loader

class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, model, data_source, train_data, train_target, batch_size, sorting_evaluations_ago, sorting_evaluations_period, bfs, prob, sumprob, epoch):
        self.model = model
        self.data_source = data_source
        self.batch_size = batch_size
        self.data = train_data
        self.data = torch.unsqueeze(self.data, 1)
        # self.data = self.data.type(torch.cuda.FloatTensor)
        self.target = train_target

        self.sorting_evaluations_period = sorting_evaluations_period   # increase it if sorting is too expensive
        self.sorting_evaluations_ago = sorting_evaluations_ago

        self.bfs = bfs
        # make bfs a update method
        self.prob = prob
        self.sumprob = sumprob
        self.epoch = epoch

        self.stop = 0
        self.iter = 0

        self.indexes = []

    def get_scores(self):
        output, feat = self.model.forward(self.data)
        criterion = nn.CrossEntropyLoss(reduce=False)
        loss = criterion(output, self.target)
        return loss

    def __iter__(self):
        while (self.stop == 0):
            self.indexes = []
            wrt_sorted = 0
            if (self.epoch > 0):
                wrt_sorted = 1
                if (self.sorting_evaluations_ago >= self.sorting_evaluations_period):
                    self.bfs = self.bfs[self.bfs[:,0].argsort()[::-1]]
                    self.sorting_evaluations_ago = 0

            stop1 = 0
            while (stop1 == 0):
                index = self.iter
                if (wrt_sorted == 1):
                    randpos = min(random.random(), self.sumprob[-1])
                    index = bisect_right(self.sumprob, randpos)  # O(log(ntraining)), cheap
                self.indexes.append(index)
                self.iter = self.iter + 1
                if (len(self.indexes) == self.batch_size) or (self.iter == len(self.data_source)):
                    stop1 = 1

            self.sorting_evaluations_ago = self.sorting_evaluations_ago + self.batch_size
            if (self.iter == len(self.data_source)):
                # traversed the whole training dataset, proceed to the next epoch
                self.stop = 1

            idxs = []
            for idx in self.indexes:
                idxs.append(int(self.bfs[idx][1]))
            
            yield idxs

    def __len__(self):
        return len(self.data_source)
    
class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size
    
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
def sorted_data_loading(model, bs, bs_test, sorting_evaluations_ago, sorting_evaluations_period, bfs, prob, sumprob, epoch, shuffle_train = True, shuffle_test = False, device = 'cpu'):
    # Define transformations for the training set, which includes normalization
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    '''
    # Load the full training set
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Define the size of the validation set
    validation_size = 10000
    train_size = len(full_train_dataset) - validation_size

    # Split the dataset into training and validation sets
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size])

    indices = train_dataset.indices
    train_data = train_dataset.dataset.data[indices].to(device)
    train_target = train_dataset.dataset.train_labels[indices].to(device)
    '''
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)

    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)

    train_dataset = CustomDataset(X_train, y_train)
    train_data = X_train
    train_target = y_train

    validation_dataset = CustomDataset(X_val, y_val)

    test_dataset = CustomDataset(X_test, y_test)

    sampler = RandomSampler(model, train_dataset, train_data, train_target, bs, sorting_evaluations_ago, sorting_evaluations_period, bfs, prob, sumprob, epoch)

    batch_sampler = BatchSampler(sampler, bs, True)

    # Create DataLoaders for each set
    # attach the indices to the images itself
    train_loader = DataLoader(dataset = train_dataset, batch_sampler = batch_sampler, num_workers = 0)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size = bs_test, shuffle = shuffle_test)

    # Load the test set
    # test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size = bs_test, shuffle = shuffle_test)

    return train_dataset, validation_dataset, test_dataset, train_loader, validation_loader, test_loader

def CCE_loss_fn(output, targets):
    if not torch.is_tensor(targets):
        targets = torch.from_numpy(targets)

    return (-(output + 1e-5).log() * F.one_hot(targets, num_classes = 10)).sum(dim=1).mean()

def CCE_losses_fn(output, targets):
    if not torch.is_tensor(targets):
        targets = torch.from_numpy(targets)

    return (-(output + 1e-5).log() * F.one_hot(targets, num_classes = 10)).sum(dim=1)

def val_fn(model, input, targets):
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    
    if not torch.is_tensor(targets):
        targets = torch.from_numpy(targets)
    model.eval()
    output = model(input)
    test_loss = CCE_loss_fn(output, targets)

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = torch.mean(torch.eq(torch.argmax(output, dim=1), targets), dtype = torch.float)
    
    return test_loss, test_acc
    
def test(model='cnn', num_epochs=50, bs_begin=16, bs_end=16, fac_begin=100, fac_end=100, pp1 = 0, pp2 = 0, alg=1, adapt_type = 0, irun = 1):
    # Load the dataset
    print("Loading data...")

    device = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")

    network = Net().to(device)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    if (alg == 1):  # adadelta with default parameters as given in lasagne
        optimizer = optim.Adadelta(network.parameters(), lr=1.0, rho=0.95, eps=1e-06)
        algname = 'adadelta'
    if (alg == 2):  # adam with default parameters as given in lasagne
        optimizer = optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        algname = 'adam'

    # Finally, launch the training loop.
    print("Starting training...")

    filename = "pytorch_data/" + algname + "_{}_{}_{}_{}_{}_{}".format(irun, bs_end, pp1, pp2, fac_begin, fac_end) + ".txt"

    mult_bs = math.exp(math.log(bs_end/bs_begin)/num_epochs)
    mult_fac = math.exp(math.log(fac_end/fac_begin)/num_epochs)

    bfs = []                    # to store last known loss for each datapoint
    ntraining = len(X_train)    # number of training datapoints
    bfs = np.ndarray((ntraining, 2))
    l = 0
    for l in range(0, ntraining):
        bfs[l][0] = 1e+10
        bfs[l][1] = int(l)

    prob = [None] * ntraining     # to store probabilies of selection, prob[i] for i-th ranked datapoint
    sumprob = [None] * ntraining  # to store sum of probabilies from 0 to i-th ranked point

    sorting_evaluations_period = 100   # increase it if sorting is too expensive
    sorting_evaluations_ago = 2 * sorting_evaluations_period

    lastii = 0
    curii = 0
    
    start_time0 = time.time()
    wasted_time = 0         # time wasted on computing training and validation losses/errors
    best_validation_error = 1e+10
    best_predicted_test_error = 1e+10
    myfile=open(filename, 'w+')

    # We iterate over epochs:
    for epoch in range(num_epochs):
        start_time = time.time()
        fac = fac_begin * math.pow(mult_fac, epoch)
        if (adapt_type == 0):   # linear
            bs = bs_begin + (bs_end - bs_begin)*(float(epoch)/float(num_epochs-1))
        if (adapt_type == 1):   # exponential
            bs = bs_begin * math.pow(mult_bs, epoch)
        bs = int(math.floor(bs))

        if (fac == 1):
            train_dataset, validation_dataset, test_dataset, train_loader, validation_loader, test_loader = regular_data_loading(bs = bs)
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                network.train()
                output = network(inputs)
                loss = CCE_loss_fn(output, targets)
                loss.backward()
                optimizer.step()
        else:
            mult = math.exp(math.log(fac) / ntraining)

            for i in range(0, ntraining):
                if (i == 0):    prob[i] = 1.0
                else:           prob[i] = prob[i - 1] / mult
            psum = sum(prob)
            prob =[v / psum for v in prob]
            for i in range(0, ntraining):
                if (i == 0):    sumprob[i] = prob[i]
                else:           sumprob[i] = sumprob[i-1] + prob[i]

            train_dataset, validation_dataset, test_dataset, train_loader, validation_loader, test_loader = sorted_data_loading(network, bs, 500, sorting_evaluations_ago, sorting_evaluations_period, bfs, prob, sumprob, epoch)
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                network.train()
                output = network(inputs)
                losses = CCE_losses_fn(output, targets)
                meanloss = losses.mean()
                meanloss.backward()
                optimizer.step()
                i = 0
                indice = train_loader.batch_sampler.sampler.indexes
                for idx in indice:
                    bfs[idx][0] = losses[i] # update loss for corresponding datapoint, rely on the computed index, so index cannot be wrapped into a sampler that is invisible to the training loop
                    i = i + 1

                #if (1):
                curii = curii + len(indice)

                if (pp1 > 0):
                    if (curii - lastii > ntraining / pp1):
                        lastii = curii
                        stopp = 0
                        iii = 0
                        bs_here = 500
                        maxpt = int(len(X_train) * pp2)
                        while (stopp == 0):
                            indexes = []
                            stop1 = 0
                            while (stop1 == 0):
                                index = iii
                                indexes.append(index)
                                iii = iii + 1
                                if (len(indexes) == bs_here) or (iii == maxpt):
                                    stop1 = 1

                            if (iii == maxpt):
                                stopp = 1

                            idxs = []
                            for idx in indexes:
                                idxs.append(int(bfs[idx][1]))

                            '''
                            inputs = train_loader.dataset.dataset.train_data[idxs]
                            targets = train_loader.dataset.dataset.train_labels[idxs]
                            '''
                            inputs = train_loader.dataset.data[idxs]
                            targets = train_loader.dataset.labels[idxs]
                            
                            # inputs = torch.unsqueeze(inputs, 1).to(torch.float)
                            network.eval()
                            output = network(inputs)
                            losses = CCE_losses_fn(output, targets)
                            i = 0
                            for idx in indexes:
                                bfs[idx][0] = losses[i]
                                i = i + 1


        if (1): # otherwise report time only

            train_loader = DataLoader(dataset = train_dataset, batch_size = bs, shuffle = False)

            validation_loader = DataLoader(dataset=validation_dataset, batch_size = bs, shuffle = False)

            test_loader = DataLoader(dataset=test_dataset, batch_size = bs, shuffle = False)

            start_time_wasted0 = time.time()
            # a full pass over the training data:
            train_err = 0
            train_acc = 0
            train_batches = 0

            for batch in train_loader:
                inputs, targets = batch
                err, acc = val_fn(network, inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1

            # a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in validation_loader:
                inputs, targets = batch
                err, acc = val_fn(network, inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            cur_valid_error = 100 - val_acc / val_batches * 100

            if (cur_valid_error < best_validation_error):
                best_validation_error = cur_valid_error
                test_err = 0
                test_acc = 0
                test_batches = 0
                for batch in test_loader:
                    inputs, targets = batch
                    err, acc = val_fn(network, inputs, targets)
                    test_err += err
                    test_acc += acc
                    test_batches += 1

                best_predicted_test_error = 100 - test_acc / test_batches * 100

            start_time_wasted1 = time.time()
            epoch_wasted_time = start_time_wasted1 - start_time_wasted0
            wasted_time = wasted_time + epoch_wasted_time
            # Then we print the results for this epoch:
            print("Epoch {} of {}".format(epoch + 1, num_epochs))
            curtime = time.time()
            epoch_learning_time = curtime - start_time - epoch_wasted_time
            epoch_total_time = curtime - start_time
            total_learning_time = curtime - start_time0 - wasted_time
            total_time = curtime - start_time0
            print("epoch learning time {:.3f}s, epoch total time {:.3f}s, total learning time {:.3f}s, total time {:.3f}s".format(epoch_learning_time, epoch_total_time, total_learning_time, total_time))
            print("{}_{:.6f}".format(bs,fac))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  train error:\t\t{:.3f} %".format(100 - train_acc / train_batches * 100))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation error:\t\t{:.3f} % , test error:\t\t{:.3f} % ".format(cur_valid_error,best_predicted_test_error))
            myfile.write("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(epoch + 1, train_err / train_batches,
                                        val_err / val_batches, cur_valid_error, best_predicted_test_error, total_learning_time, total_time))

        else:
             print("Epoch {} of {} took {:.3f}s, total {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time, time.time() - start_time0))

    network.eval()
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in test_loader:
        inputs, targets = batch
        err, acc = val_fn(network, inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        100 - test_acc / test_batches * 100))
    
def main():
    num_epochs = 50  # maximum number of epochs
    alg = 1         # 1 - AdaDelta, 2 - Adam, see function test for more details

    #bs_begin = 64   # batch size at epoch 0
    #bs_end = 64     # batch size at epoch 'num_epochs'
    #fac_begin = 1   # selection pressure at at epoch 0
    #fac_end = 1     # selection pressure at at 'num_epochs'
    adapt_type = 1  # 0 - linear, 1 - exponential change of batch size from bs_begin to bs_end as a function of epoch index

    run_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    alg_vals = [1, 2]
    pp_scenarios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    bs_vals = [64]
    for irun in run_vals:
        for bs in bs_vals:
            bs_begin = bs
            bs_end = bs
            for scenario in pp_scenarios:
                if (scenario == 1):  fac_begin = 1;      fac_end = 1;       pp1 = 0;    pp2 = 0;
                if (scenario == 2):  fac_begin = 1.01;   fac_end = 1.01;    pp1 = 0;    pp2 = 0;

                if (scenario == 3):  fac_begin = 1e+2;   fac_end = 1e+2;    pp1 = 0;    pp2 = 0;
                if (scenario == 4):  fac_begin = 1e+2;   fac_end = 1.01;    pp1 = 0;    pp2 = 0;
                if (scenario == 5):  fac_begin = 1e+8;   fac_end = 1e+8;    pp1 = 0;    pp2 = 0;
                if (scenario == 6):  fac_begin = 1e+8;   fac_end = 1.01;    pp1 = 0;    pp2 = 0;

                if (scenario == 7):  fac_begin = 1e+2;   fac_end = 1e+2;    pp1 = 0.5;    pp2 = 1.0;
                if (scenario == 8):  fac_begin = 1e+2;   fac_end = 1.01;    pp1 = 0.5;    pp2 = 1.0;
                if (scenario == 9):  fac_begin = 1e+8;   fac_end = 1e+8;    pp1 = 0.5;    pp2 = 1.0;
                if (scenario == 10):  fac_begin = 1e+8;   fac_end = 1.01;    pp1 = 0.5;    pp2 = 1.0;


                for alg in alg_vals:
                    test('cnn', num_epochs, bs_begin, bs_end, fac_begin, fac_end, pp1, pp2, alg, adapt_type, irun)

if __name__ == '__main__':
    main()
