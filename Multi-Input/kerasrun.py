import kerasLstm
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def write_mse(row, socre):
    f = open("stock.txt","wb+")
    f.write(row, score)
    f.close()

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    #plt.show()
    plt.savefig(pic_name)

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

    seq_len = 5

    print('> Loading data.....')
    for row in range(80, 680,5):

        X_train, y_train, X_test, y_test = kerasLstm.load_data('./stock.csv', seq_len, True, row)
        A_train, b_train, A_test, b_test = kerasLstm.load_data('./stock.csv', seq_len, True, row)
        print('> Data Loaded. Compiling...')
        model = kerasLstm.build_model([1,5, 100, 1])

        model.fit(
            X_train,
            y_train,
            batch_size=1,
            epochs=30,
            validation_split=( X_test, y_test))

    #predicted = kerasLstm.predict_sequences_multiple(model, X_test, seq_len, 20)
        predicted = kerasLstm.predict_sequences_full(model, X_test, seq_len)
    #predicted = kerasLstm.predict_point_by_point(model, X_test)

        #print('Training duration (s):', time.time() - global_start_time)

    #rmse = np.sqrt(((predicted - b_test) ** 2).mean(axis=0))
        score = mean_squared_error(predicted, b_test)
        write_mse(row, score)
        pic_name = ["stock"] + row
    #Mean_value = np.mean(predicted, axis= 0)
        #print("MSE: %f" % score)
    #print("MEAN %f" % Mean_value )
    #x = np.array(A_test)
    #xx = pd.DataFrame({'B': [x]})
    #yy = pd.Series(predicted)
    #res = pd.ols(y=yy,x=xx)
    #print("RES: %f" % res)

    #plot_results_multiple(predicted, b_test, 20)

        plot_results(predicted, b_test, pic_name)
