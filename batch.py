from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import re
from collections import Counter
from string import punctuation
import torch

def batch_data(words, sequence_length, batch_size):
   
    words = np.array(words, dtype=np.int)
    
    n_batches = len(words)//(batch_size*sequence_length)
    words = words[:batch_size*sequence_length*n_batches]
    
    words = words.reshape((batch_size, -1))
    feature_tensor = []
    target_tensor = []
    for n in range(0, words.shape[1], batch_size):
        x = words[:, n:n+sequence_length]
        y = np.zeros((1, batch_size), dtype=int)
        try:
            y[-1] = words[:, n+sequence_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], words[:, 0]
        feature_tensor.append(x)
        target_tensor.append(y)

    tensor_x = torch.stack([torch.Tensor(i) for i in feature_tensor]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in target_tensor])
    
    data = TensorDataset(tensor_x.long(), tensor_y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1)
    return data_loader


def batch_data1(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    # create Tensor datasets
    
    words = np.array(words, dtype=np.int)
    
    n_batches = len(words)//(batch_size*sequence_length)
     # only full batches
        
    # print(words.shape)
    words = words[:batch_size*sequence_length*n_batches]
    # print(words.shape)
    
    words = words.reshape((batch_size, -1))
    # print(words.shape)
    feature_tensor = []
    target_tensor = []
    for n in range(0, words.shape[1], batch_size):
        x = words[:, n:n+sequence_length]
        # The targets, shifted by one
        y = np.zeros((1, batch_size), dtype=int)
        try:
            # y[:, :-1], y[:, -1] = x[:, 1:], words[:, n+sequence_length]
            # y[:-1] = x[:, 1:]
            y[-1] = words[:, n+sequence_length]
        except IndexError:
            # y[-1] = words[:, 0]
            y[:, :-1], y[:, -1] = x[:, 1:], words[:, 0]        
        feature_tensor = x
        target_tensor = y
        # feature_tensor.append(x)
        # target_tensor.append(y)
        # if len(feature_tensor) == 0:
            #feature_tensor = x
            #target_tensor = y
        #else:
            # print(feature_tensor)
            # print(x)
            #feature_tensor = np.concatenate((feature_tensor , x), axis=0)
            #target_tensor = np.concatenate((target_tensor , y), axis=0)
        # target_tensor.append(y)
        # feature_tensor = x
        # target_tensor.append(y)
    
    target_tensor = target_tensor.flatten()

    # tensor_x = torch.stack([torch.Tensor(i) for i in feature_tensor]) # transform to torch tensors
    # tensor_y = torch.stack([torch.Tensor(i) for i in target_tensor])
    
    # [wordsfor word in words]
    # exit
    # feature_tensor = np.zeros((batch_size, sequence_length), dtype=int)
    # target_tensor = np.zeros((1, batch_size), dtype=int)
    
    # feature_tensor[0, -len(words):] = np.array(words)[:sequence_length]
    # target_tensor[-len(words):] = np.array(words)[sequence_length:]
    # feature_tensor = words[:sequence_length]
    # target_tensors
    
    # print(torch.from_numpy(feature_tensor).size())
    # print(torch.from_numpy(target_tensor).size())
    
    data = TensorDataset(torch.from_numpy(feature_tensor), torch.from_numpy(target_tensor))
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

    # print(words.shape)
    # print(torch.from_numpy(feature_tensor).size())
    # print(target_tensor)
    # return a dataloader
    return data_loader

def test_getitem_1d(words, n_batches):
    words = np.array(words, dtype=np.int)
    t = torch.randn(892110)
    l = torch.randn(892110)
    source = TensorDataset(words, l)
    data_loader = torch.utils.data.DataLoader(source, batch_size=n_batches)
    return data_loader

test_text = range(892110)
t_loader = batch_data(test_text, sequence_length=5, batch_size=128)
# t_loader = test_getitem_1d(test_text,10)
print("----------------------------------------------------------------------------------")
print(len(t_loader.dataset))
for batch_i, (inputs, labels) in enumerate(t_loader, 1):
    print("----inputs----",batch_i)
    # inputs = inputs.flatten()
    print(inputs[0])
    labels = labels.flatten()
    print(labels)
    print(labels.shape)

# data_iter = iter(t_loader)
# sample_x, sample_y = data_iter.next()

# print(sample_x.shape)
# print(sample_x)
# print()
# print(sample_y.shape)
# print(sample_y)

# my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
# my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

# print(my_x)

# tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
# tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

# print(tensor_x)