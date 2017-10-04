import pandas as pd
import numpy as np

if __name__ == "__main__":
    bld_names = ['4057_Foege_WD7', '4057_Foege_WE7','4057_Foege_WF7']
    
    minlen = 1000000
    load_v = []
    for bld_name in bld_names:
        data = pd.read_csv('filtered/' + bld_name + '.csv')
        datetime = data['date-time'].values
        load = data['load'].values
        minlen = min(minlen, load.size)
        
        load_v.append(load)
    
    load_combined = np.zeros((minlen))
    for load in load_v:
        load_combined += load[-minlen:]
    
    datetime_combined = datetime[-minlen:]
    
    d = dict({'date-time' : datetime_combined, 'load' : load_combined})
    df = pd.DataFrame(d)    
    df.to_csv('combined/' + bld_name + '.csv', sep=',', index = False)
  
        
    

