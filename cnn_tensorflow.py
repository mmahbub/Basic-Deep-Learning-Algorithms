import numpy as np
import random
import matplotlib.pyplot as plt 
import time
import sys
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, losses
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from numpy import unravel_index
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd

def reshape_data(trainX, testX, channels_first = False, flatten = True):
    if channels_first:
        trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
        testX = testX.reshape((testX.shape[0], 1, 28, 28))
    elif flatten:
        trainX = trainX.reshape((trainX.shape[0], 784))
        testX = testX.reshape((testX.shape[0], 784))
    else:
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
    return trainX, testX

def one_hot_encoder(trainY, testY):
    # creating instance of one-hot-encoder
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
    trainY = enc.fit_transform(trainY.reshape(-1, 1)).toarray()
    testY = enc.transform(testY.reshape(-1, 1)).toarray()
    inv_testY = enc.inverse_transform(testY)
    return trainY, testY, inv_testY

def minmax_scaling(trainX, testX):
    mm_scaler = preprocessing.MinMaxScaler()
    train_shape = trainX.shape
    trainX = mm_scaler.fit_transform(trainX.flatten().reshape(-1, 1)).reshape(train_shape)
    test_shape = testX.shape
    testX = mm_scaler.transform(testX.flatten().reshape(-1, 1)).reshape(test_shape)
    return trainX, testX

class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def plot_time_loss(train_time, train_loss, val_time, val_loss):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_time, train_loss, label = 'Train')
    plt.plot(val_time, val_loss, label = 'Validation ')
    plt.title("Time-Loss Plot for Fashion MNIST Dataset")
    plt.xlabel('Time(seconds)')
    plt.ylabel('Loss')
    plt.legend(loc = 'best')
    plt.show()


def plot_epoch_acc(history):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history['accuracy'], label = 'Train')
    plt.plot(history.history['val_accuracy'], label = 'Validation ')
    plt.title("Epoch-Accuracy Plot for Fashion MNIST Dataset")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()


def plot_epoch_loss(history):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history['loss'], label = 'Train') 
    plt.plot(history.history['val_loss'], label = 'Validation ') 
    plt.title("Epoch-Loss Plot for Fashion MNIST Dataset")
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(loc='best') 
    plt.show()


def plot_confusion_matrix(trainY, inv_testY, predY, task_number ):
    labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    cm = confusion_matrix(inv_testY, predY)

    df_cm = pd.DataFrame(cm, labels, labels)
    plt.figure(figsize=(16,12))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap="OrRd", fmt='g') # font size
    plt.title("Confusion Matrix for "+ "Task #" + str(task_number) )
    plt.show()

def task1(trainX, trainY, testX, lr, epochs, batch_size):
    # Task 1
    cb = TimeHistory()
    print('Keras Solution')
    model = models.Sequential()
    model.add(layers.Dense(784, activation='tanh'))
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    H = model.fit(trainX, trainY, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[cb])
    predY = model.predict(testX)
    predY = [np.argmax(pred) for pred in list(predY)]
    return H, cb.times, predY

def task2(trainX, trainY, testX, lr, epochs, batch_size):
    # Task 2
    cb = TimeHistory()
    print('Keras Solution')
    model = models.Sequential()
    model.add(layers.Conv2D(40, (5, 5), activation='relu', padding='valid'))#, input_shape=(5, 5, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])    
    H = model.fit(trainX, trainY, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[cb])
    predY = model.predict(testX)
    predY = [np.argmax(pred) for pred in list(predY)]
    return H, cb.times, predY

def task3(trainX, trainY, testX, lr, epochs, batch_size):
    # Task 3
    cb = TimeHistory()
    print('Keras Solution')
    model = models.Sequential()
    model.add(layers.Conv2D(48, (3, 3), activation='relu', padding='valid'))#, input_shape=(5, 5, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(layers.Conv2D(96, (3, 3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    H = model.fit(trainX, trainY, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[cb])
    model.summary()
    predY = model.predict(testX)
    predY = [np.argmax(pred) for pred in list(predY)]
    return H, cb.times, predY

def task4(trainX, trainY, testX,lr, epochs, batch_size):
    # Task 4
    cb = TimeHistory()
    print('Keras Solution')
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    H = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, callbacks=[cb], validation_split=0.2)
    model.summary()
    predY = model.predict(testX)
    predY = [np.argmax(pred) for pred in list(predY)]
    return H, cb.times, predY

def task5(trainX, trainY, testX, epochs, batch_size, latent_dim):    
    cb = TimeHistory()
    
    input_shape = (trainX.shape[1], trainX.shape[1], 1)
     
    en_inputs = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Conv2D(filters=32,kernel_size=3,activation='relu',strides=2, padding='same')(en_inputs)
    x = layers.Conv2D(filters=64,kernel_size=3,activation='relu',strides=2, padding='same')(x)
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_sdv = layers.Dense(latent_dim, name='z_sdv')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_sdv])

    # build encoder model
    encoder = models.Model(en_inputs, [z_mean, z_sdv, z], name='encoder')
    encoder.summary()

    # build decoder model
    de_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(de_inputs)
    x = layers.Reshape((shape[1], shape[2], shape[3]))(x)
    x = layers.Conv2DTranspose(filters=64,kernel_size=3,activation='relu',strides=2,padding='same')(x)
    x = layers.Conv2DTranspose(filters=32,kernel_size=3,activation='relu',strides=2,padding='same')(x)


    outputs = layers.Conv2DTranspose(filters=1,kernel_size=3,activation='sigmoid',padding='same',name='decoder_output')(x)

    # instantiate decoder model
    decoder = models.Model(de_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(en_inputs)[2])
    vae = models.Model(en_inputs, outputs, name='vae')
    
    loss = calc_loss(en_inputs, outputs, trainX.shape[1], z_mean, z_sdv, mse=False, bce=True)
     
    vae.add_loss(loss)
    vae.compile(optimizer='adam', metrics=['accuracy'])

    H = vae.fit(trainX,epochs=epochs,batch_size=batch_size, validation_data=(testX, None), callbacks=[cb])
    
    vae.summary()
    return encoder, decoder, H, cb.times


def calc_loss(inputs, outputs, image_size, z_mean, z_sdv, mse=False, bce=True): 
    if mse:
        reconstruction_loss = losses.mse(K.flatten(inputs), K.flatten(outputs))
    else:
        reconstruction_loss = losses.binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_sdv - K.square(z_mean) - K.exp(z_sdv)
    kl_loss = -0.5*K.sum(kl_loss, axis=-1)
    loss = K.mean(reconstruction_loss + kl_loss)
    
    return loss

def plot_results(encoder, decoder, latent_dim, max_range):
    digit_size = 28
    figure = np.zeros((2*digit_size,5*digit_size))

    for i in range(0, 2):
        for j in range(0,5):
            z_sample = np.array([ np.random.uniform(low=-max_range, high=max_range, size=(latent_dim,))])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 25))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()   
    
def sampling(args):
    z_mean, z_sdv = args
    return z_mean + K.exp(0.5 * z_sdv) * K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))


def main():
    command = sys.argv[1]
    ((trainX, trainY), (testX, testY)) = datasets.fashion_mnist.load_data()
    
    if command == 'Task1':
        print ("Task1")
        trainX, testX = reshape_data(trainX, testX, channels_first = False, flatten = True)
        trainY, testY, inv_testY = one_hot_encoder(trainY, testY)
        trainX, testX = minmax_scaling(trainX, testX)
        print(trainX.shape)
        print(trainY.shape)

        history, time, predY = task1(trainX, trainY, testX, 0.7, 50, 200)

        plot_time_loss(np.cumsum(time), history.history['loss'], np.cumsum(time), history.history['val_loss'])
        plot_epoch_acc(history)
        plot_epoch_loss(history)
        plot_confusion_matrix(trainY, inv_testY, predY, 1)

        print("Classification Accuracy: ", accuracy_score(inv_testY, predY))


    elif command == 'Task2':
        print ("Task2")
        trainX, testX = reshape_data(trainX, testX, channels_first = False, flatten = False)
        trainY, testY, inv_testY = one_hot_encoder(trainY, testY)
        trainX, testX = minmax_scaling(trainX, testX)
        print(trainX.shape)
        print(trainY.shape)

        history, time, predY = task2(trainX, trainY, testX, 0.07, 50, 200)

        plot_time_loss(np.cumsum(time), history.history['loss'], np.cumsum(time), history.history['val_loss'])
        plot_epoch_acc(history)
        plot_epoch_loss(history)
        plot_confusion_matrix(trainY, inv_testY, predY, 2)

        print("Classification Accuracy: ", accuracy_score(inv_testY, predY))
    
    elif command == 'Task3':
        print ("Task3")
        trainX, testX = reshape_data(trainX, testX, channels_first = False, flatten = False)
        trainY, testY, inv_testY = one_hot_encoder(trainY, testY)
        trainX, testX = minmax_scaling(trainX, testX)
        print(trainX.shape)
        print(trainY.shape)

        history, time, predY = task3(trainX, trainY, testX, 0.01, 50, 200)

        plot_time_loss(np.cumsum(time), history.history['loss'], np.cumsum(time), history.history['val_loss'])
        plot_epoch_acc(history)
        plot_epoch_loss(history)
        plot_confusion_matrix(trainY, inv_testY, predY, 3)

        print("Classification Accuracy: ", accuracy_score(inv_testY, predY))
    
    elif command == 'Task4':
        print ("Task4")
        trainX, testX = reshape_data(trainX, testX, channels_first = False, flatten = False)
        trainY, testY,inv_testY = one_hot_encoder(trainY, testY)
        trainX, testX = minmax_scaling(trainX, testX)
        print(trainX.shape)
        print(trainY.shape)
    
        history, time, predY = task4(trainX, trainY, testX, 0.1, 50, 200)

        plot_time_loss(np.cumsum(time), history.history['loss'], np.cumsum(time), history.history['val_loss'])
        plot_epoch_acc(history)
        plot_epoch_loss(history)
        plot_confusion_matrix(trainY, inv_testY, predY, 4)

        print("Classification Accuracy: ", accuracy_score(inv_testY, predY))

    else: 
        print ("Task5")
        trainX, testX = reshape_data(trainX, testX, channels_first = False, flatten = False)
        trainY, testY, inv_testY = one_hot_encoder(trainY, testY)
        trainX, testX,  = minmax_scaling(trainX, testX)        

        encoder, decoder, history, time = task5(trainX, trainY, testX, 50, 200, 8)
        
        plot_time_loss(np.cumsum(time), history.history['loss'], np.cumsum(time), history.history['val_loss'])
        plot_epoch_loss(history)
        plot_results(encoder, decoder, 8, 2.0)


main()

