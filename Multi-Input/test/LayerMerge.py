import os
import io
import warnings
import keras
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import newaxis
from keras.layers.core import  Activation, Dropout
from keras.models import Model
from keras.layers import Input, Embedding,LSTM, Dense,TimeDistributed
from keras import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import concatenate
#from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def series_to_supervised(data, n_in, n_out, dropnan=True):
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

def plot_results(predicted_data, true_data, name):
    #print (pic_name)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    #plt.show()
    plt.savefig('multi'+str(name)+'.png')

def write_mse(socre, name):
    try:
        f = io.open('multi'+str(name)+'.txt',"w",encoding='utf-8')
    except IOError:
        print("failed")
        return

    f.write(unicode(score))
    f.close()

if __name__=='__main__':

    dataset_stock = read_csv('stock1.csv', header = 0, index_col=0)
    values_stock = dataset_stock.values
    encoder = LabelEncoder()
    values_stock[:,0] = encoder.fit_transform(values_stock[:,0])
    values_stock = values_stock.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_stock = scaler.fit_transform(values_stock)
    reframed_stock = series_to_supervised(scaled_stock, 1, 1)

    dataset_gap = read_csv('gap1.csv', header = 0, index_col=0)
    values_gap = dataset_gap.values
    encoder = LabelEncoder()
    values_gap[:,0] = encoder.fit_transform(values_gap[:,0])
    values_gap = values_gap.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_gap = scaler.fit_transform(values_gap)
    reframed_gap = series_to_supervised(scaled_gap, 1, 1)

    dataset_fai = read_csv('fai1.csv', header = 0, index_col=0)
    values_fai = dataset_fai.values
    encoder = LabelEncoder()
    values_fai[:,0] = encoder.fit_transform(values_fai[:,0])
    values_fai = values_fai.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_fai = scaler.fit_transform(values_fai)
    reframed_fai = series_to_supervised(scaled_fai, 1, 1)

    values_stock = reframed_stock.values
    values_gap = reframed_gap.values
    values_fai = reframed_fai.values
    for row in range(85, 680, 5):
        n_end = row + 5
        train_stock = values_stock[:row, :]
        test_stock = values_stock[row: n_end :]
        train_X_stock, train_y_stock = train_stock[:,:-1], train_stock[:, -1]
        test_stock_X, test_stock_y = test_stock[:,:-1],test_stock[:,-1]
        train_X_stock = train_X_stock.reshape((train_X_stock.shape[0], 1, train_X_stock.shape[1]))
        test_stock_X = test_stock_X.reshape((test_stock_X.shape[0], 1, test_stock_X.shape[1]))

        train_gap = values_gap[:row, :]
        test_gap = values_gap[row:n_end :]
        train_X_gap, train_y_gap = train_gap[:,:-1], train_gap[:, -1]
        test_gap_X, test_gap_y = test_gap[:,:-1],test_gap[:,-1]
        train_X_gap = train_X_gap.reshape((train_X_gap.shape[0], 1, train_X_gap.shape[1]))
        test_gap_X = test_gap_X.reshape((test_gap_X.shape[0], 1, test_gap_X.shape[1]))

        train_fai = values_fai[:row, :]
        test_fai = values_fai[row:n_end :]
        train_X_fai, train_y_fai = train_fai[:,:-1], train_fai[:, -1]
        test_fai_X, test_fai_y = test_fai[:,:-1],test_fai[:,-1]
        train_X_fai = train_X_fai.reshape((train_X_fai.shape[0], 1, train_X_fai.shape[1]))
        test_fai_X = test_fai_X.reshape((test_fai_X.shape[0], 1, test_fai_X.shape[1]))


        stock_input = Input(shape=(train_X_stock.shape[1], train_X_stock.shape[2]), dtype='float32', name='stock_input')
        lstm_stock = LSTM(32)(stock_input)
        lstm_out_stock = Dense(1)(lstm_stock)
        stock_output = Dense(1, activation='linear', name='stock_output')(lstm_out_stock)

        gap_input = Input(shape=(train_X_gap.shape[1], train_X_gap.shape[2]),dtype='float32', name='gap_input')
        lstm_out_stock = Embedding(output_dim=1, input_dim = 10000000)(lstm_out_stock)

        merg_first = keras.layers.concatenate([lstm_out_stock, gap_input])
        lstm_out_gap = LSTM(64)(merg_first)
        lstm_out_gap = Dense(1)(lstm_out_gap)
        gap_output = Dense(1, activation='linear', name='gap_output')(lstm_out_gap)

        fai_input = Input(shape=(train_X_fai.shape[1], train_X_fai.shape[2] ),dtype='float32', name='fai_input')

        lstm_out_gap = Embedding(output_dim=1, input_dim = 10000000)(lstm_out_gap)
        merg_second = keras.layers.concatenate([lstm_out_gap, fai_input])
        lstm_fai = LSTM(128)(merg_second)
        lstm_out_fai = Dense(1)(lstm_fai)
        last_layer = Dense(32, activation='linear')(lstm_out_fai)
        main_output = Dense(1, activation='linear', name='main_output')(last_layer)

        model = Model(inputs=[stock_input, gap_input, fai_input], outputs=[stock_output, gap_output, main_output])
        model.compile(optimizer='rmsprop',
              loss={'main_output':'mse', 'stock_output':'mse', 'gap_output':'mse'},
              loss_weights={'main_output':1., 'stock_output': 0.2, 'gap_output': 0.2}
            )


        history=model.fit(x={'stock_input':train_X_stock,
                   'gap_input': train_X_gap,
                   'fai_input': train_X_fai},
               y= {'main_output':train_y_fai, 'stock_output':train_y_stock, 'gap_output':train_y_gap},
                 epochs=30, batch_size=1)
        
        predicted = model.predict({'stock_input':train_X_stock,'gap_input': train_X_gap,'fai_input': train_X_fai})

        predicted= predicted[2:3]
        predicted = np.reshape(predicted,(-1,1))
        result = predicted[:5]

        
        score = mean_squared_error(result, test_stock_y)
        name = []
        name = row
        write_mse(score, name)
        print("MSE: %f" % score)
        plot_results(result, test_stock_y, name)
