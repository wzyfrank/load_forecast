import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Kern(t, ti, h):
    delta_t = t - ti
    k = 1.0 / np.sqrt(2 * np.pi * np.square(h) ) * np.exp(-0.5 * np.square(delta_t) / np.square(h) )
    return k


def getS(t_series, h, N):
    S = np.zeros((N, N))
    S_sum = np.zeros((N))
    for i in range(N):
        for j in range(N):
            S[i, j] = Kern(t_series[i], t_series[j], h)
            S_sum[j] += S[i, j]

    for i in range(N):
        for j in range(N):
            S[i, j] = S[i, j] / S_sum[j]
    
    return S

def load_correction(loads, d, Td, S, coef):
    N = 9 * Td
    load_test = loads[(d-1)*Td: (d+8)*Td]
    load_max = np.max(load_test)
    load_test = load_test / load_max #normalize data
    load_pred = np.dot(S, load_test)
    
    ############ confidence interval ##########################################
    df = np.trace(S)
    error = load_test - load_pred
    error_sum = np.sum(np.square(error))
    MSE = error_sum / (N - df)

    Var = MSE * np.dot(S, S.T)
    
    LB = np.zeros(N)
    UB = np.zeros(N)
    
    xaxis = range(N)
    for i in range(N):
        s = np.sqrt(MSE + Var[i,i])
        LB[i] = load_pred[i] - coef * s
        UB[i] = load_pred[i] + coef * s
        
    
    ############################ plot the results #############################
    
    plt.figure(figsize=(18,10))
    plt.plot(xaxis, load_test, 'r')
    plt.plot(xaxis, load_pred, 'g')
    plt.plot(xaxis, LB, 'g--')
    plt.plot(xaxis, UB, 'g--')   
    
    part_day = int(Td / 4) # one fourth of day as threshold
    for i in range(part_day,N- part_day):
        if load_test[i] < LB[i] or load_test[i] > UB[i]:
            #print(i)
            load_test[i] = load_pred[i]
    
    plt.plot(xaxis, load_test, 'y')
    plt.show()
    
    load_test = load_test * load_max
    return load_test[Td:Td*8]
    
def Kernel_clean(bld_name):    
    Td = 96
    # duration : a week, 96 * (7+2) data points
    N = Td * 9
    # t series
    t_series = np.arange(N) + 1
    t_series = t_series / N
    # h parameter
    h = 6 * (1/ N)
    # calculate S
    S = getS(t_series, h, N)
            
    # 1st time Kernel filtering
    coef1 = 3 # coefficient of condifence interval
    # import the load data
    data = pd.read_csv('cleaned/' + bld_name + '_cleaned.csv')
    loads = data['load'].values
    timestamp = data['date-time'].values

    data_cleaned = loads
    d = 1
    while(d * Td + 8*Td <= loads.size):
        data_cleaned[d*Td:(d+7)*Td] = load_correction(loads, d, Td, S, coef1)
        d += 7   
    
    d = int(loads.size/Td - 8)
    data_cleaned[d*Td:(d+7)*Td] = load_correction(loads, d, Td, S, coef1)
    
    print('second time cleaning')
    # 2nd time Kernel filtering
    loads = data_cleaned
    coef2 = 2.5
    d = 1
    while(d * Td + 8*Td <= loads.size):
        data_cleaned[d*Td:(d+7)*Td] = load_correction(loads, d, Td, S, coef2)
        d += 7   
    
    d = int(loads.size/Td - 8)
    data_cleaned[d*Td:(d+7)*Td] = load_correction(loads, d, Td, S, coef2)    
    
    
    dt = dict({'date-time' : timestamp, 'load' : data_cleaned})
    df = pd.DataFrame(dt)    
    df.to_csv('filtered/' + bld_name + '.csv', sep=',', index = False)

    

    
            