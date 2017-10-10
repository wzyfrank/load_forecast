import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import outlier


def BuildTimeMap(bld_name):
    # time(integer) to load (kW) map
    load_map = dict()
    # read the .csv file
    loads_list = pd.read_csv('parsed/' + bld_name + '.csv', sep = ',')
    N_lines = len(loads_list.index) 
    last_load = -1.0 # load of last effective entry
    last_time = -1 # time of last effective entry
    first_time = -1
    
    load_base = 0.0
    
    for i in range(N_lines):
        load_time = loads_list.iloc[i, 0]
        if load_time != 'timestamp':
            d = datetime.datetime.strptime(load_time, '%Y-%m-%dT%H:%M:%S')
            t_load = int(time.mktime(d.timetuple()))
            # redundent data points
            if t_load in load_map.keys():
                continue
            
            # abnormal data (too large)
            if float(loads_list.iloc[i, 1]) > 1000000.0:
                continue
            
            # missing entry, use the last one 
            elif loads_list.iloc[i, 1] == 'None':
                load_map[t_load] = last_load
            
            # meter stuck, current reading equals last reading
            elif float(loads_list.iloc[i, 1]) + load_base == last_load:
                # get start time
                t_start = last_time
                t_list = [t_start]
                energy_start = last_load
                
                # while meter is stuck, record timestamps 
                while float(loads_list.iloc[i, 1]) + load_base == last_load:
                    cur_t = loads_list.iloc[i, 0]
                    d_cur = datetime.datetime.strptime(cur_t, '%Y-%m-%dT%H:%M:%S')
                    t_cur = int(time.mktime(d_cur.timetuple()))
                    t_list.append(t_cur)
                    i += 1
                    # skip header row and missing entry
                    while loads_list.iloc[i, 0] == 'timestamp' or loads_list.iloc[i, 1] == 'None':
                        i += 1
                
                # get end time
                end_t = loads_list.iloc[i, 0]
                d_end = datetime.datetime.strptime(end_t, '%Y-%m-%dT%H:%M:%S')
                t_end = int(time.mktime(d_end.timetuple())) 
                t_list.append(t_end)
                energy_end = float(loads_list.iloc[i, 1]) + load_base
                if energy_end < energy_start:
                    load_base += 100000.0
                    energy_end += 100000.0
                
                # interpolate
                x = np.asarray([t_start, t_end])
                y = np.asarray([energy_start, energy_end])
                f = interpolate.interp1d(x, y)
                
                # record to map
                t_list = np.asarray(t_list)
                new_energy = f(t_list)
                for i in range(t_list.size):
                    load_map[t_list[i]] = new_energy[i]
                
                # update last_load
                last_load = energy_end
                last_time = t_end
                if first_time == -1:
                    first_time = last_time
                
            else:
                # normal entry, check whether load in list is valid
                if(float(loads_list.iloc[i, 1]) + load_base < last_load):
                    load_base += 100000.0
                
                cur_load = load_base + float(loads_list.iloc[i, 1])
                load_map[t_load] = cur_load
                last_load = cur_load
                last_time = t_load
                if first_time == -1:
                    first_time = last_time
              
    return (load_map, first_time, last_time)


def get_StartTime(first_t):
    d = datetime.datetime.fromtimestamp(first_t) 
    month_d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31};
    d_max = month_d[d.month]
    if d.day == d_max:
        start_time = str(d.year) + '-' + str(d.month+1) + '-0' + str(1) + 'T00:00:00'
    else:
        start_time = str(d.year) + '-' + str(d.month) + '-' + str(d.day+1) + 'T00:00:00'
    return start_time


def get_EndTime(last_t):
    d = datetime.datetime.fromtimestamp(last_t) 
    # month_d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31};
    end_time = str(d.year) + '-' + str(d.month) + '-' + str(d.day) + 'T00:00:00'
    return end_time


def GenkWLoad(bld_name, figure_on):
    # build the timestamp -> kWh map
    (load_map, first_time, last_time) = BuildTimeMap(bld_name)
    start_time = get_StartTime(first_time)
    end_time = get_EndTime(last_time)
    
    start_t = datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    start_t = int(time.mktime(start_t.timetuple()))
    end_t = datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
    end_t = int(time.mktime(end_t.timetuple()))
    
    # the timestamp of kW readings
    new_time = np.arange(start_t,end_t,900)
    newload_map = dict()
    kW_load = []
    
    
    for t in new_time:
        # t in kWh map
        if t in load_map.keys():
            newload_map[t] = load_map[t]          
        elif t - 900 * 96 in newload_map.keys() and t - 900 * 97 in newload_map.keys():
            newload_map[t] = newload_map[t-900] + newload_map[t-900*96] - newload_map[t-900*97]
        elif t - 900 * 192 in newload_map.keys() and t - 900 * 193 in newload_map.keys():
            newload_map[t] = newload_map[t-900] + newload_map[t-900*192] - newload_map[t-900*193]
        else:
            newload_map[t] = newload_map[t-900] + newload_map[t-900] - newload_map[t-900*2]
        
        # record to kW reading map
        if(t > start_t):
            # change from 15min kWh to kW
            if newload_map[t] - newload_map[t-900] > -2500.0 and newload_map[t] - newload_map[t-900] < 2500.0: 
                kW_load.append(4 * (newload_map[t] - newload_map[t-900]))
            else:
                kW_load.append(kW_load[-96])
    
    # figure
    new_time = new_time[1:]
    if figure_on:
        plt.figure(figsize=(18,12))
        plt.plot(new_time, kW_load)
        plt.show()  
    
    # output to a .csv file
    timestamp = []
    for t in new_time:
        timestamp.append(datetime.datetime.fromtimestamp(t))
    dt = dict({'date-time' : timestamp, 'load' : kW_load})
    df = pd.DataFrame(dt) 
    # write output to csv file
    df.to_csv('cleaned/' + bld_name + '_cleaned.csv', sep=',', index = False)



if __name__ == "__main__":
    ############# inputs #####################
    # bulding name
    bld_name = '1008_EE_CSE_WB3_accum'
    # select the data range
    start_time = get_StartTime(bld_name) 
    # start_time = '2014-12-01T00:00:00'    
    end_time = get_EndTime(bld_name) 
    #end_time = '2014-12-01T00:00:00'   
    # read data, make sure the time sequence is correct
    print('Creating load profile')
    figure_on = True
    GenkWLoad(bld_name, start_time, end_time, figure_on)
    
    print('Filtering abnormal data...')
    outlier.Kernel_clean(bld_name, figure_on)
    
    

    
    
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
        
        