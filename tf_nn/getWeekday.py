import pandas as pd
import numpy as np
import datetime

def getWeekdayload(bld_name):
    data = pd.read_csv('data/' + bld_name + '.csv')
    timestamp = data['date-time'].values
    
    load = data['load'].values
    # append the last value in load
    last = float(load[-1])
    load = np.concatenate((load, [last]))
    load_len = load.size
    load_weekday = []
    t = 0
    while t < load_len:
        #d = datetime.datetime.strptime(timestamp[t], '%Y-%m-%d %H:%M:%S')
        d = datetime.datetime.strptime(timestamp[t], '%m/%d/%Y %H:%M')
        # check whether the date is weekday
        if d.isoweekday() in range(1, 6):
            load_weekday.append(load[t:t+96])
        t += 96
    
    load_weekday = np.asarray(load_weekday)
    load_weekday = load_weekday.flatten()
    return load_weekday

    

    