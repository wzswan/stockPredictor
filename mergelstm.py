#from __future__ import division

import os
import time
import warnings
import numpy as np
import keras
from numpy import newaxis
from keras.layers.core import  Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.layers import Input, Embedding,LSTM, Dense



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide messy Tensorflow warings
warnings.filterwarnings("ignore") # Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    # divide the data into parts by seq_len, so get n matrix

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)
    # select 90% data as training data
    rest = result[:-5]
    row =  rest.shape[0]
    train = result[:int(row), :]
    # random sorted training data
    #np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    # 10% as test data
    x_test = result[int(row):, : -1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalise_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalise_window)
    return normalised_data



def build_model(layers):

    main_input = Input(shape=(layers[0],), dtype='float64', name='main_input')

    embed= Embedding(output_dim=512, input_dim=10000, input_length=10)(main_input)

    lstm_out= LSTM(layers[1])(embed)

    auxiliary_output = Dense(layers[2], activation='linear', name='aux_output')(lstm_out)

    auxiliary_input = Input(shape=(layers[3],), name='aux_input')

    merge1 = keras.layers.concatenate([lstm_out, auxiliary_input])
    embed2= Embedding(output_dim=512, input_dim=10000, input_length=10)(merge1)
    lstm_out2= LSTM(layers[4])(embed2)

    auxiliary_output2 = Dense(layers[5], activation='linear', name='aux_output2')(lstm_out2)

    auxiliary_input2 = Input(shape=(layers[6],), name='aux_input2')
    merge2 = keras.layers.concatenate([lstm_out2, auxiliary_input2 ])

    x = Dense(layers[7], activation='linear')(merge2)

    main_output = Dense(layers[8], activation='linear', name='main_output')(x)

    inputs=[main_input, auxiliary_input, auxiliary_input2]
    outputs=[main_output, auxiliary_output, auxiliary_output2]
    model= Model(inputs,
                 outputs)
    start = time.time()
    model.compile(optimizer='rmsprop',
                    loss={'main_output': 'mse', 'aux_output':'mse', 'aux_output2':'mse'},
                    loss_weights={'main_output': 1., 'aux_output': 0.2, 'aux_output2': 0.2})
    print("> Compilation Time :", time.time() - start)
    return model



def predict_sequences_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1],predicted[-1], axis=0)
    return predicted
