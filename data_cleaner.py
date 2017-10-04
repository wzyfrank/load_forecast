import pandas as pd
import datetime
import time

#t_str = '2015-04-07 19:11:21'
#d = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
def read_rawData(bld_name, start_time, end_time):
    ##########################################################
    # time(integer) to load (kW) map
    load_map = dict()
    
    # read the .csv file
    loads_list = pd.read_csv(bld_name + '.csv', sep = ',')
    N_lines = len(loads_list.index)
    for i in range(N_lines):
        load_time = loads_list.iloc[i, 0]
        if load_time != 'timestamp':
            d = datetime.datetime.strptime(load_time, '%Y-%m-%dT%H:%M:%S')
            t_load = int(time.mktime(d.timetuple()))
            # check if 
            if loads_list.iloc[i, 1] == 'None':
                load_map[t_load] = 0.0
                #print("missing load data at time" + load_time)
            else:
                
                load_map[t_load] = float(loads_list.iloc[i, 1])
     
    
    start_t = datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    start_t = int(time.mktime(start_t.timetuple()))
    end_t = datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
    end_t = int(time.mktime(end_t.timetuple()))
    
    
    # create the data frame
    df = pd.DataFrame([], [], ['timestamp', 'load'])
    row = 0
    t = start_t
    while(t < end_t):
        cur_time = datetime.datetime.fromtimestamp(t)
        if(t in load_map.keys() and load_map[t] > 0.0):
            df.loc[row] = [cur_time, load_map[t]]
        else:
            t_yes = t - 86400
            if t_yes not in load_map:
                t_yes = t - 900
            load_yes = load_map[t_yes]
            df.loc[row] = [cur_time, load_yes] # timestamp missing or abnormal value
            load_map[t] = load_yes # update the load map
        row += 1
        
        t += 900 # move to next 15min interval
     
    # write output to csv file
    df.to_csv('cleaned/' + bld_name + '_cleaned.csv', sep=',', index = False)
    
def get_StartTime(bld_name):
    loads_list = pd.read_csv(bld_name + '.csv', sep = ',')
    first_time = loads_list.iloc[0, 0]
    d = datetime.datetime.strptime(first_time, '%Y-%m-%dT%H:%M:%S')
    month_d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31};
    d_max = month_d[d.month]
    if d.day == d_max:
        start_time = str(d.year) + '-' + str(d.month+1) + '-0' + str(1) + 'T00:00:00'
    else:
        start_time = str(d.year) + '-' + str(d.month) + '-' + str(d.day+1) + 'T00:00:00'
    return start_time
'''
if __name__ == "__main__":
    ############# inputs #####################
    # bulding name
    bld_name = '1008_EE_CSE_WC3'
    # select the data range
    start_time = get_StartTime(bld_name)    
    end_time = '2012-07-06T00:00:00'
    
    # read data, make sure the time sequence is correct
    read_rawData(bld_name, start_time, end_time)
    
    # read cleaned data, apply Kernel filter
    data = pd.read_csv('cleaned/' + bld_name + '_cleaned.csv')
    timestamp = data['timestamp'].values
    # call Kernel filter data cleansing function
    load_cleaned = outlier2.Kernel_clean(bld_name)
    d = dict({'date-time' : timestamp, 'load' : load_cleaned})
    df = pd.DataFrame(d)    
    df.to_csv('filtered/' + bld_name + '.csv', sep=',', index = False)
'''
        
        