import getWeekday
import predict_util
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans

def forecast_kMeans_NN(bld_name, load_weekday, n_train, n_lag, n_clusters):
    n_days = int(load_weekday.size / T)
    MAPE_sum_nn = 0.0
    RMSPE_sum_nn = 0.0
    for curr_day in range(n_train + n_lag, n_days-1):
        y_train = np.zeros((n_train, T))
        X_train = np.zeros((n_train, T * n_lag))
        row = 0
        for train_day in range(curr_day - n_train, curr_day):
            y_train[row,:] = load_weekday[train_day * T : train_day * T + T]
            X_train[row,:] = load_weekday[train_day * T - n_lag * T: train_day * T]
            row += 1
            
        # building test data
        X_test = load_weekday[curr_day*T - n_lag*T: curr_day*T]
        y_test = load_weekday[curr_day*T: curr_day *T + T]
        
        # n_clusters = 5
        kmeans = KMeans(n_clusters, random_state=0)
        kmeans.fit(y_train.T)
        labels = list(kmeans.labels_)
        
        y_nn = np.zeros((T))
        # cluster NN
        for c in range(n_clusters):
            cluster_idx = [i for i,x in enumerate(labels) if x == c]
            cluster_train = []
            for k in range(n_lag):
                cluster_sub = [a+k*T for a in cluster_idx]
                cluster_train.extend(cluster_sub)            
            X_train_c = X_train[:, cluster_train]
            y_train_c = y_train[:, cluster_idx]            
            X_test_c = X_test[cluster_train]
           
            nn = MLPRegressor(hidden_layer_sizes = (50, 50), activation = 'relu', max_iter = 10000)        
            nn.fit(X_train_c, y_train_c)
            y_nn_c = nn.predict(X_test_c)
            y_nn[cluster_idx] = y_nn_c    
        
        # statistics of Neural Network
        MAPE_nn = predict_util.calMAPE(y_test, y_nn)
        MAPE_sum_nn += MAPE_nn
        RMSPE_nn = predict_util.calRMSPE(y_test, y_nn)
        RMSPE_sum_nn += RMSPE_nn
        
        
    days_sample = n_days - 1 - n_train - n_lag
    MAPE_avg_nn = MAPE_sum_nn / days_sample
    RMSPE_avg_nn = RMSPE_sum_nn / days_sample
    return (MAPE_avg_nn, RMSPE_avg_nn)


if __name__ == "__main__":
    bld_names = ['combined/1008_EE_CSE', 'combined/1108_Chem', 'combined/1111_Fluke', 'combined/1126_Meany', 'combined/1143_McMahon', 'combined/1147_Haggett', 'combined/1158_McCarty', 'combined/1163_Port_Bay', 'combined/1195_Hec_Ed', 'combined/1201_Gowen', 'combined/1275_Pool', 'combined/1306_Physics', 'combined/1316_BAEEC', 'combined/1357_Fish_Sc', 'combined/4057_Foege']
    T = 96

    n_train = 50
    n_lag = 5
    n_clusters = 2
    
    nn_MAPE = []
    nn_RMSPE = []
    
    for bld_name in bld_names:
        load_weekday = getWeekday.getWeekdayload(bld_name)
        (MAPE_avg_nn, RMSPE_avg_nn) = forecast_kMeans_NN(bld_name, load_weekday, n_train, n_lag, n_clusters)
        nn_MAPE.append(MAPE_avg_nn)
        nn_RMSPE.append(RMSPE_avg_nn)
    
    d = dict({'bld_name' : bld_names, 'nn_MAPE' : nn_MAPE, 'nn_RMSPE' : nn_RMSPE})
    df = pd.DataFrame(d)    
    df.to_csv('NN_24cluster_forecast_results.csv', sep=',', index = False)