import predict_util
import numpy as np
import pandas as pd
import getWeekday
from sklearn.neural_network import MLPRegressor


def forecast_bagging_NN(bld_name, load_weekday, n_train, n_lag, bag_num):
    n_days = int(load_weekday.size / T)
    MAPE_sum_nn = 0.0
    RMSPE_sum_nn = 0.0    
    bag_size = int(n_train * 0.6)

    for curr_day in range(n_train + n_lag, n_days-1):
        # bagging parameter
        train_start = curr_day - n_train
        for i in range(bag_num):
            # random sampling
            sample_day = np.random.randint(n_train, size = bag_size)
            sample_day += train_start
            
            # building training data
            y_train = np.zeros((bag_size, T))
            X_train = np.zeros((bag_size, T * n_lag))
            for row in range(bag_size):
                y_train[row,:] = load_weekday[sample_day[row] * T : sample_day[row] * T + T]
                X_train[row,:] = load_weekday[sample_day[row] * T - n_lag * T: sample_day[row] * T]
                    
            # building test data
            X_test = np.reshape(load_weekday[curr_day*T - n_lag*T: curr_day*T], (1, -1))
            y_test = load_weekday[curr_day*T: curr_day *T + T]
            
            nn = MLPRegressor(hidden_layer_sizes = (50, 50), activation = 'relu', max_iter = 10000)        
            nn.fit(X_train, y_train)
            
            y_nn = nn.predict(X_test)
            
            MAPE_nn = predict_util.calMAPE(y_test, y_nn)
            MAPE_sum_nn += MAPE_nn
            RMSPE_nn = predict_util.calRMSPE(y_test, y_nn)
            RMSPE_sum_nn += RMSPE_nn    
        
    days_sample = n_days - 1 - n_train - n_lag
    MAPE_avg_nn = MAPE_sum_nn / (days_sample * bag_num)
    RMSPE_avg_nn = RMSPE_sum_nn / (days_sample * bag_num)
    return (MAPE_avg_nn, RMSPE_avg_nn)

    
if __name__ == "__main__":
    bld_names = ['combined/1008_EE_CSE', 'combined/1108_Chem', 'combined/1111_Fluke', 'combined/1126_Meany', 'combined/1143_McMahon', 'combined/1147_Haggett', 'combined/1158_McCarty', 'combined/1163_Port_Bay', 'combined/1195_Hec_Ed', 'combined/1201_Gowen', 'combined/1275_Pool', 'combined/1306_Physics', 'combined/1316_BAEEC', 'combined/1357_Fish_Sc', 'combined/4057_Foege']
    T = 96

    n_train = 50
    n_lag = 5
    n_clusters = 2
    bag_num = 10
    
    nn_MAPE = []
    nn_RMSPE = []
    
    for bld_name in bld_names:
        print(bld_name)
        load_weekday = getWeekday.getWeekdayload(bld_name)
        (MAPE_avg_nn, RMSPE_avg_nn) = forecast_bagging_NN(bld_name, load_weekday, n_train, n_lag, bag_num)
        nn_MAPE.append(MAPE_avg_nn)
        nn_RMSPE.append(RMSPE_avg_nn)
    
    d = dict({'bld_name' : bld_names, 'nn_MAPE' : nn_MAPE, 'nn_RMSPE' : nn_RMSPE})
    df = pd.DataFrame(d)    
    df.to_csv('NN_10bags_forecast_results.csv', sep=',', index = False)
    
    

    

    