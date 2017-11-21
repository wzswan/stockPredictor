import mergelstm
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
#import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()



if __name__=='__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 20

    print('> Loading data.....')

    X_train, y_train, X_test, y_test = mergelstm.load_data('./week/w1/faip1.csv', seq_len, True)
    X_train_gap, y_train_gap, X_test_gap, y_test_gap = mergelstm.load_data('./week/w1/gapw1.csv', seq_len, True)
    X_train_stock, y_train_stock, X_test_stock, y_test_stock = mergelstm.load_data('./week/w1/stockw1.csv', seq_len, True)


    validation_data=({'fai_input': X_test,
                      'gap_input': X_test_gap,
                      'stock_input':X_test_stock},
                      {'stock_val': y_test_stock})
    """
    training_data = ({'fai_input': X_train,
                      'gap_input': X_train_gap,
                      'stock_input':X_train_stock},
                      {'stock_val': y_train_stock})
    """
    print('> Data Loaded. Compiling...')

    model = mergelstm.build_model([5,32,3,5,128,3,5,64,3])

    model.fit({'main_input': X_train, 'aux_input':  X_train_gap, 'aux_input2': X_train_stock},
                {'main_output': y_train, 'aux_output': y_train_gap, 'aux_output2':y_train_stock},
                nb_epoch=epochs, batch_size=32, shuffle=False,verbose = 1,
                #validation_split= 0.005)
                validation_data=validation_data)




    predicted = mergelstm.predict_sequences_full(model, X_test, seq_len)


    print('Training duration (s):', time.time() - global_start_time)

    rmse = np.sqrt(((predicted - b_test) ** 2).mean(axis=0))
    score = mean_squared_error(predicted, b_test)
    print("MSE: %f" % score)
    plot_results(predicted, b_test)
