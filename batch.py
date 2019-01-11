from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import re
from collections import Counter
from string import punctuation
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

data_dir = './data/Seinfeld_Scripts.txt'
data_dir_1 = './data/corpus_banda.txt'

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()
    return data

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

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

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    
    dict_token = {}
    dict_token['.'] = '||PERIOD||'
    dict_token[','] = '||COMMA||'
    dict_token['"'] = '||QUOTATION_MARK||'
    dict_token[';'] = '||SEMICOLON||'
    dict_token['!'] = '||EXCLAMATION_MARK||'
    dict_token['?'] = '||QUESTION_MARK||'
    dict_token['('] = '||LEFT_PARENTHESES||'
    dict_token[')'] = '||RIGHT_PARENTHESES||'
    dict_token['-'] = '||DASH||'
    dict_token['?'] = '||QUESTION_MARK||'
    dict_token['\n'] = '||RETURN||'
    # dict_token[':'] = ' ||COLON|| '
    # print(dict_token)
    return dict_token

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

def generator(words, sequence_length, batch_size):
    index = 0
    x = np.zeros((batch_size, sequence_length, len(words)), dtype=np.bool)
    y = np.zeros((batch_size, len(words)), dtype=np.bool)

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {(ii): word for ii, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return (vocab_to_int, int_to_vocab)

def batch_data_final(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    words = np.array(words, dtype=np.int)
    
    n_batches = len(words)//(batch_size*sequence_length)
    words = words[:batch_size*sequence_length*n_batches]
    # words = words.reshape((batch_size, -1))
    x = np.array([words[n:n+sequence_length] for n in range(0, len(words)) if (n+sequence_length) <= len(words) ])
    # x = x.reshape((len(x), -1))
    y = np.array([words[n] for n in range(sequence_length, len(words))], dtype=np.int)
    y = np.concatenate((y, [words[0]]), axis=0)
    # y = y.reshape((-1, len(y)))
    # x = np.asarray(x, dtype=np.int)
    # print(words)
    # print(x.shape)
    # print(y.shape)
    # print(x)
    # print(y)
    # print(torch.Tensor(y).size())
    data = TensorDataset(torch.Tensor(x).long(), torch.Tensor(y).long())
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    # print(data_loader.dataset)
    return data_loader

def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)


class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_length = sequence_length
        
        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.fc1 = nn.Linear(128, output_size)
        self.sig = nn.Softmax(dim=1)
        # self.hidden = self.init_hidden()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        
        # nn_input = nn_input.t()
        # batch_size = nn_input.size(1)
    
        # TODO: Implement function 
        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.dropout(self.embedding(nn_input))

        # print(embeds.shape)
        # print(embeds.view(len(nn_input), self.window_length, -1).shape)
        
        lstm_out, hidden = self.lstm(embeds.view(len(nn_input), self.window_length, -1), hidden)
  
        # stack up lstm outputs
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # out = self.fc1(out)
        
        # sigmoid function
        # sig_out = self.sig(out)
        sig_out = out

        # print(sig_out.shape)
        
        sig_out = sig_out.view(batch_size, -1, self.output_size)
        # print(sig_out.size())
        # get last batch
        sig_out = sig_out[:, -1]
        # print(sig_out.size())
        # print(sig_out.size())
        # sig_out = sig_out[:,-1] # get last batch of labels

        sig_out = F.log_softmax(sig_out, dim=1)

        # return one batch of output word scores and the hidden state
        return sig_out, hidden    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available

        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches):
    batch_losses = []

    previousLoss = np.Inf
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)

    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        print("epoch ",epoch_i)
        for batch_i, (inputs, labels) in enumerate(t_loader, 1):
            n_batches = len(t_loader.dataset) // batch_size
            if(batch_i > n_batches):
                break
            
            # hidden = tuple([each.data for each in hidden])

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            hidden = repackage_hidden(hidden)
            rnn.zero_grad()
            try:
                # get the output from the model
                output, hidden = rnn(inputs, hidden)        
            except RuntimeError:
                raise
            # print(labels)
            loss = criterion(output, labels)
            loss.backward()
            # optimizer.zero_grad()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(rnn.parameters(),  5)
            for p in rnn.parameters():
                p.data.add_(-learning_rate, p.grad.data)

            optimizer.step()

            for p in rnn.parameters():
                p.data.add_(-learning_rate, p.grad.data)

            batch_losses.append(loss.item())
            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                average_loss = np.average(batch_losses)
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(epoch_i, n_epochs, average_loss))
                if average_loss <= previousLoss:
                    previousLoss = average_loss
                    save_model('./save/trained_rnn_temp', trained_rnn)            
                    batch_losses = []
                    print('Model Trained and Saved')
            
            # if batch_i % show_every_n_batches == 0:
                # print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(epoch_i, n_epochs, np.average(batch_losses)))
                # batch_losses = []
    # returns a trained rnn
    return rnn

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
    
SPECIAL_WORDS = {'PADDING': '<PAD>'}
text = load_data(data_dir)
text = text[81:]
lines = text.split('\n')
token_dict = token_lookup()
for key, token in token_dict.items():
    text = text.replace(key, ' {} '.format(token))
text = text.lower()
# text = text.split()
text = text.split()
vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
int_text = [vocab_to_int[word] for word in text]

print(len(lines))
print(len(text))

int_text_text = int_text
# print(vocab_to_int)

num_epochs = 5
learning_rate = 0.01
vocab_size = len(vocab_to_int) + 1
output_size = len(vocab_to_int) + 1
embedding_dim = 750
hidden_dim = 300
n_layers = 2
show_every_n_batches = 1000

sequence_length = 140
batch_size = 200

rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print(rnn)

# test_text = range(892110)
t_loader = batch_data_final(int_text_text, sequence_length, batch_size)
print("totla-batch-size ",len(t_loader.dataset))
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)
# t_loader = test_getitem_1d(test_text,10)
print("----------------------------------------------------------------------------------")
# print(len(t_loader.dataset))
# for batch_i, (inputs, labels) in enumerate(t_loader, 1):
    # print("----inputs----",batch_i)
    # inputs = inputs.flatten()
    # print(inputs)
    # labels = labels.flatten()
    # print(labels)
    # print(labels.shape)

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