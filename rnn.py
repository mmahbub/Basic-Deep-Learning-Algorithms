# import libraries
import sys
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt 
import pickle
from numpy import array
from pickle import dump, load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
def training_sequence_generator(filename, window_size, stride):
    # load text
    raw_text = load_doc(filename)
    
    # clean
    tokens = raw_text.split()
    raw_text = ' '.join(tokens)

    # organize into sequences of characters
    length = window_size
    sequences = list()
    for i in range(length, len(raw_text), stride):
        # select sequence of tokens
        seq = raw_text[i-length:i+1]
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    print('Sequences:', sequences[:50])
    # save sequences to file
    out_filename = 'char_sequences.txt'
    save_doc(sequences, out_filename)
    
def array_generator(in_filename):
    # load
    raw_text = load_doc(in_filename)
    lines = raw_text.split('\n')

    # integer encode sequences of characters
    chars = sorted(list(set(raw_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    sequences = list()
    for line in lines:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)

    # vocabulary size
    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)

    # separate into input and output
    sequences = array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
    X = array(sequences)
    y = to_categorical(y, num_classes=vocab_size)

    # save the mapping
    dump(mapping, open('mapping.pkl', 'wb'))
    return X, y, vocab_size

def build_LSTM(X_shape, vocab_size, hidden_state):
    # define model
    model = Sequential()
    model.add(LSTM(hidden_state, input_shape=(X_shape[1], X_shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def build_RNN(X_shape, vocab_size, hidden_state):
    # define model
    model = Sequential()
    model.add(SimpleRNN(hidden_state, input_shape=(X_shape[1], X_shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# generate a sequence of characters with a language model
def new_sequence_generator(model, mapping, window_size, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=window_size, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = encoded.reshape(1, window_size, 48)
        # predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += char
    return in_text

def translate_encoded_sequence(sample, mapping):
    
    text = ''
    for s in sample:
        out_char = ''
        for char, index in mapping.items():
            if index == np.argmax(s):
                out_char = char
                break
        text+=out_char
       
    return text

class GenerateSequenceCallback(callbacks.Callback):
    
    # initialize callback for printing random sequences during training
    def __init__(self, data, window_size):
        # load training data
        self.training_data = data
        # load character mappings
        self.mapping = load(open('mapping.pkl', 'rb'))
        self.window_size = window_size

    
    def on_epoch_end(self, epoch, logs=None):
        # generate random sequences every 5 epochs
        if epoch % 5 == 4:
            for k in range(0,5):
                line = random.choice(self.training_data)
                print('Random Sequence', str(k+1), new_sequence_generator(self.model, self.mapping, self.window_size, translate_encoded_sequence(line, self.mapping), 10))

                
def train_model(model, X, y, n_epochs, window_size):
    
    # trains given model for n_epochs; returns history
    cb = GenerateSequenceCallback(X, window_size)
    H = model.fit(X, y, epochs=n_epochs, verbose=2, callbacks=[cb])
    
    return H     


def main():
    
    epochs = 100
    text_file = sys.argv[1]
    model = sys.argv[2]
    hidden_state = int(sys.argv[3])
    window_size = int(sys.argv[4])
    stride = int(sys.argv[5])
    
    print('Generating Training Sequence')
    training_sequence_generator(text_file, window_size, stride)
    print('Generating Arrays for Training')
    X, y, vocab_size = array_generator('char_sequences.txt')
    
    if model == 'lstm':
        print('Building LSTM Model')
        model_LSTM = build_LSTM(X.shape, vocab_size, hidden_state)
        print('Training LSTM Model 100 Epochs')
        train_model(model_LSTM, X, y, epochs, window_size)
        
    else:
        print('Building RNN Model')
        model_RNN = build_RNN(X.shape, vocab_size, hidden_state)
        print('Training RNN Model 100 Epochs')
        train_model(model_RNN, X, y, epochs, window_size)
    
main()
        
