import pandas as pd
import datetime
import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import outlier
import callRPCA

#t_str = '2015-04-07 19:11:21'
#d = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
def BuildTimeMap(bld_name):
    # time(integer) to load (kW) map
    time_list = []
    load_list = []
    # read the .csv file
    loads_list = pd.read_csv('energy/' + bld_name + '.csv', sep = ',')
    N_lines = len(loads_list.index) 
    last_load = float(loads_list.iloc[0, 1])
    load_base = 0.0
    
    for i in range(N_lines):
        load_time = loads_list.iloc[i, 0]
        if load_time != 'timestamp':
            d = datetime.datetime.strptime(load_time, '%Y-%m-%dT%H:%M:%S')
            t_load = int(time.mktime(d.timetuple()))
            # bad data point
            if loads_list.iloc[i, 1] != 'None' and loads_list.iloc[i, 1] != last_load:
                if(float(loads_list.iloc[i, 1]) + load_base < last_load):
                    load_base += 100000.0
                
                time_list.append(t_load)
                cur_load = load_base + float(loads_list.iloc[i, 1])
                load_list.append(cur_load)
                last_load = cur_load
              
    return (time_list, load_list)


def GenkWLoad(bld_name, start_time, end_time):
    ##########################################################

    start_t = datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    start_t = int(time.mktime(start_t.timetuple()))
    start_t = start_t - 900
    end_t = datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
    end_t = int(time.mktime(end_t.timetuple()))
    
    (time_list, load_list) = BuildTimeMap(bld_name)
    
    x = np.asarray(time_list)
    y = np.asarray(load_list)
    
    f = interpolate.interp1d(x, y)
    newx = np.arange(start_t,end_t,900)
    newy = f(newx)
    
    newx = newx[1:]
    newy = newy[1:] - newy[0:-1]
    
    plt.figure(figsize=(18,12))
    plt.plot(newx, newy)
    plt.show()  
    timestamp = []
    for t in newx:
        timestamp.append(datetime.datetime.fromtimestamp(t))
    dt = dict({'date-time' : timestamp, 'load' : newy})
    df = pd.DataFrame(dt) 
    # write output to csv file
    df.to_csv('cleaned/' + bld_name + '_cleaned.csv', sep=',', index = False)


    
def get_StartTime(bld_name):
    loads_list = pd.read_csv('energy/' + bld_name + '.csv', sep = ',')
    first_time = loads_list.iloc[0, 0]
    d = datetime.datetime.strptime(first_time, '%Y-%m-%dT%H:%M:%S')
    month_d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31};
    d_max = month_d[d.month]
    if d.day == d_max:
        start_time = str(d.year) + '-' + str(d.month+1) + '-0' + str(1) + 'T00:00:00'
    else:
        start_time = str(d.year) + '-' + str(d.month) + '-' + str(d.day+1) + 'T00:00:00'
    return start_time


def get_EndTime(bld_name):
    loads_list = pd.read_csv('energy/' + bld_name + '.csv', sep = ',')
    N_lines = len(loads_list.index)
    last_time = loads_list.iloc[N_lines-1, 0]
    d = datetime.datetime.strptime(last_time, '%Y-%m-%dT%H:%M:%S')
    # month_d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31};
    end_time = str(d.year) + '-' + str(d.month) + '-' + str(d.day) + 'T00:00:00'
    return end_time


if __name__ == "__main__":
    ############# inputs #####################
    # bulding name
    bld_name = '1008_EE_CSE_WC3_accum'
    # select the data range
    # start_time = get_StartTime(bld_name) 
    start_time = '2014-01-12T00:00:00'    
    #end_time = get_EndTime(bld_name) 
    end_time = '2014-12-01T00:00:00'   
    # read data, make sure the time sequence is correct
    print('Creating load profile')
    GenkWLoad(bld_name, start_time, end_time)
    
    print('Filtering abnormal data...')
    # outlier.Kernel_clean(bld_name)
    
    callRPCA.callRPCA(bld_name)
    
    
    '''
    # read cleaned data, apply Kernel filter
    data = pd.read_csv('cleaned/' + bld_name + '_cleaned.csv')
    timestamp = data['timestamp'].values
    # call Kernel filter data cleansing function
    load_cleaned = outlier2.Kernel_clean(bld_name)
    d = dict({'date-time' : timestamp, 'load' : load_cleaned})
    df = pd.DataFrame(d)    
    df.to_csv('filtered/' + bld_name + '.csv', sep=',', index = False)
    '''
        
        