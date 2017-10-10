import numpy as np
import pandas as pd
import datetime
import time
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

def load_correction(loads, d, Td, S, coef, figure_on):
    N = 9 * Td
    Tp = int(Td * 3 / 4)
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
    if figure_on:
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
    
    if figure_on:
        plt.plot(xaxis, load_test, 'y')
        plt.show()
    
    load_test = load_test * load_max
    return load_test[Td-Tp:Td*8+Tp]
    

def Prepare_load(bld_names):
    if len(bld_names) == 1:
        data = pd.read_csv('cleaned/' + bld_names[0] + '_cleaned.csv')
        timestamp = data['date-time'].values  
        loads = np.asarray(data['load'].values)      
        return(timestamp, loads) 
        
    t_start = 0
    t_end = 999999999999999
    
    for bld_name in bld_names:
        data = pd.read_csv('cleaned/' + bld_name + '_cleaned.csv')
        timestamp = data['date-time'].values    
        time_start = timestamp[0]
        time_end = timestamp[-1]
        d_start = datetime.datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S')
        d_end = datetime.datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S')
        t_start = max(t_start, int(time.mktime(d_start.timetuple())))
        t_end = min(t_end, int(time.mktime(d_end.timetuple())))
    
    ds = datetime.datetime.fromtimestamp(t_start)
    de = datetime.datetime.fromtimestamp(t_end)
    
    str_start = str(ds)
    str_end = str(de)
    
    load_size = int((t_end - t_start) / 900) + 1
    loads = np.zeros((load_size))
    for bld_name in bld_names:
        data = pd.read_csv('cleaned/' + bld_name + '_cleaned.csv')
        timestamp = data['date-time'].values  
        load = np.asarray(data['load'].values)
        s_idx = np.where(timestamp == str_start)[0][0]
        e_idx = np.where(timestamp == str_end)[0][0]
        loads += load[s_idx:e_idx+1]
        
    timestamp = []
    t = t_start
    while t < t_end+1:
        timestamp.append(datetime.datetime.fromtimestamp(t))
        t += 900
    
    return(timestamp, loads)
    
    
def Kernel_clean(bld_names, figure_on):    
    Td = 96
    Tp = int(Td * 3 / 4)
    # duration : a week, 96 * (7+2) data points
    N = Td * 9
    # t series
    t_series = np.arange(N) + 1
    t_series = t_series / N
    # h parameter
    h = 6 * (1/ N)
    # calculate S
    S = getS(t_series, h, N)
            
    # prepare the load data
    (timestamp, loads) = Prepare_load(bld_names)
    
    # 1st time Kernel filtering    
    data_cleaned = loads
    coef1 = 3 # coefficient of condifence interval
    d = 1
    while(d * Td + 8*Td <= loads.size):
        data_cleaned[d*Td-Tp:(d+7)*Td+Tp] = load_correction(loads, d, Td, S, coef1, figure_on)
        d += 7   
    
    d = int(loads.size/Td - 8)
    data_cleaned[d*Td-Tp:(d+7)*Td+Tp] = load_correction(loads, d, Td, S, coef1, figure_on)
    
    
    # 2nd time Kernel filtering
    loads = data_cleaned
    coef2 = 2.5
    d = 1
    while(d * Td + 8*Td <= loads.size):
        data_cleaned[d*Td-Tp:(d+7)*Td+Tp] = load_correction(loads, d, Td, S, coef2, figure_on)
        d += 7   
    
    d = int(loads.size/Td - 8)
    data_cleaned[d*Td-Tp:(d+7)*Td+Tp] = load_correction(loads, d, Td, S, coef2, figure_on)    


    # 3rd time Kernel filtering
    loads = data_cleaned
    coef2 = 2.5
    d = 1
    while(d * Td + 8*Td <= loads.size):
        data_cleaned[d*Td-Tp:(d+7)*Td+Tp] = load_correction(loads, d, Td, S, coef2, figure_on)
        d += 7   
    
    d = int(loads.size/Td - 8)
    data_cleaned[d*Td-Tp:(d+7)*Td+Tp] = load_correction(loads, d, Td, S, coef2, figure_on) 
    
    # write filtered data to .csv file
    dt = dict({'date-time' : timestamp, 'load' : data_cleaned})
    df = pd.DataFrame(dt)    
    df.to_csv('filtered/' + bld_names[0] + '.csv', sep=',', index = False)


'''
if __name__ == "__main__":
    bld_names = ['1008_EE_CSE_WA3_accum', '1008_EE_CSE_WB3_accum', '1008_EE_CSE_WC3_accum']
    Prepare_load(bld_names)
'''
    
            