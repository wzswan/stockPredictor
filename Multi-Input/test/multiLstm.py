#from __future__ import division

import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide messy Tensorflow warings
warnings.filterwarnings("ignore") # Hide messy Numpy warnings

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(),list()
    for i in range(n_in, 0 ,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1,i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis =1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace = True)
    return agg

def load_data(filename, seq_len, normalise_window, row):
    dataset = read_csv('input2.csv', header = 0, index_col=0)
    #dataset_stock = read_csv('stock1.csv', header = 0, index_col=0)
    values = dataset.values
    values_stock = dataset_stock.values
    encoder = LabelEncoder()
    values[:,2] = encoder.fit_transform(values[:,2])
    values_stock[:,0] = encoder.fit_transform(values_stock[:,0])
    values = values.astype('float32')
    values_stock = values_stock.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_stock = scaler.fit_transform(values_stock)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed_stock = series_to_supervised(scaled_stock, 1, 1)

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)
    # select 90% data as training data
    #row = 674
    row_end = row + 5
    train = result[:row, :]
    test = result[row:row_end :]
    # random sorted training data
    #np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    # 10% as test data
    x_test = test[:, : -1]
    y_test = test[:, -1]

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
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))
    #model.add(Activation('relu'))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))
    #model.add(Activation('relu'))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
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
