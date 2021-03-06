import mergelstm
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it is correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main run thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 20

    print('> Loading data.....')

    X_train, y_train, X_test, y_test = mergelstm.load_data('stockData.csv', seq_len, True)
    W_train, s_train, W_test, s_test = mergelstm.losd_data('faiData.csv', seq_len, True)
    A_train, b_train, A_test, b_test = mergelstm.load_data('stockData.csv', seq_len, True)
    print('> Data Loaded. Compiling...')
    input1 = y_train
    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = s_train
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    averaged = keras.layers.Average()([x1, x2])

    out = keras.layer.Dense(4)(averaged)

    model = mergelstm.build_model([1,20, 100, 1])

    model.fit(
        X_train,
        y_train,
        batch_size=1,
        nb_epoch=epochs,
        validation_split=0.005)

    predicted = mergelstm.predict_sequences_multiple(model, X_test, seq_len, 20)
    #predicted = mergelstm.predict_sequences_full(model, X_test, seq_len)
    #predicted = mergelstm.predict_point_by_point(model, X_test)

    print('Training duration (s):', time.time() - global_start_time)
    """
    rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
    score = mean_squared_error(predicted, y_test)
    print("MSE: %f" % score)
    """
    plot_results_multiple(predicted, b_test, 20)

    #plot_results(predicted, b_test)
