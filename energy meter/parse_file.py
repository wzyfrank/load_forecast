import pandas as pd


def parsefile(bld_name):
    # read the file
    loads_list = pd.read_csv('energy/' + bld_name + '.csv', sep = ',')
    N_lines = len(loads_list.index) 
    
    # skip the first rows
    i = 0
    while True:
        if loads_list.iloc[i, 0] == 'timestamp':
            break
        i = i + 1
    
    # extract the kWh reading the following rows
    timestamp = []
    kWh_reading = []
    
    name_idx = bld_name.find('accum')
    bld_kWh = bld_name[:name_idx] + 'kWH'
    kWh_idx = 0
    while i < N_lines:
        if loads_list.iloc[i, 0] == 'timestamp':
            
            if loads_list.iloc[i, 1] == bld_kWh:
                kWh_idx = 1
            elif loads_list.iloc[i, 2] == bld_kWh:
                kWh_idx = 2
            elif loads_list.iloc[i, 3] == bld_kWh:
                kWh_idx = 3
            elif loads_list.iloc[i, 4] == bld_kWh:
                kWh_idx = 4
        
        timestamp.append(loads_list.iloc[i, 0])
        kWh_reading.append(loads_list.iloc[i, kWh_idx])
        i += 1
    
    # write to a .csv file
    dt = dict({'date-time' : timestamp, 'kWh_load' : kWh_reading})
    df = pd.DataFrame(dt) 
    # write output to csv file
    df.to_csv('parsed/' + bld_name + '.csv', sep=',', index = False, header = False)

