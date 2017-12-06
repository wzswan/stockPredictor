import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy import newaxis, concatenate
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_squared_error
from pandas import read_csv, DataFrame,concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential

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
    row = 673
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


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

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
A_train, b_train, A_test, b_test = load_data('./stock.csv', seq_len=20, normalise_window=True)
dataset = read_csv('input3.csv', header = 0, index_col=0)
#dataset_stock = read_csv('stock.csv', header = 0, index_col=0)
values = dataset.values
#values_stock = dataset_stock.values
encoder = LabelEncoder()
values[:,1] = encoder.fit_transform(values[:,1])
#values_stock[:,] = encoder.fit_transform(values_stock[:,])
values = values.astype('float32')
#values_stock = values_stock.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#scaled_stock = scaler.fit_transform(values_stock)
reframed = series_to_supervised(scaled, 1, 1)
#reframed_stock = series_to_supervised(scaled_stock, 1 , 1)
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis = 1, inplace=True)
print(reframed.head())
#print(reframed_stock.head())

values = reframed.values
#values_stock = reframed_stock.values_stock
n_train_hours = 673

train = values[:n_train_hours, :]
#train_stock = values_stock[:n_train_hours,:]

test = values[n_train_hours:, :]
#test_stock = values_stock[n_train_hours:, :]

train_X, train_y = train[:,:-1], train[:, -1]
#train_stock_X, train_stock_y = train_stock[:,:-1],train_stock[:, -1]

test_X, test_y = test[:, :-1], test[:,-1]
#test_stock_X, test_stock_y = test_stock[:,:-1], test_stock[:,-1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#model.add(LSTM(input_shape=(layers[1], layers[0]),output_dim=layers[1],return_sequences=True))
model.add(Dropout(0.2))
    #model.add(Activation('relu'))
model.add(LSTM(100,return_sequences=False))
model.add(Dropout(0.2))
    #model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation("linear"))

start = time.time()

model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
print("> Compilation Time :", time.time() - start)
history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat =  model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#print test_X
print yhat
inv_yhat = concatenate((yhat, test_X[:,1:]), axis=1)
#print inv_yhat
#inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis =1)
#inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]




rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#score = mean_squared_error(inv_y, b_test)
#print("MSE: %f" % score)
print('Test RMSE: %.3f' % rmse)
plot_results(inv_y, inv_yhat)

#b_test = b_test.reshape((len(b_test), 1))
#inv_real = concatenate((b_test, A_test[:, 1:]), axis=1)
#inv_real = inv_real[:,0]

mse = sqrt(mean_squared_error(b_test, yhat))
print('Real MSE: %.3f' % mse)
plot_results(b_test, yhat)
