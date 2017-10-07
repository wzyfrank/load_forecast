import r_pca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def callRPCA(bld_name):
    data = pd.read_csv('cleaned/' + bld_name + '_cleaned.csv')
    loads = data['load'].values
    timestamp = data['date-time'].values

    n_days = int(loads.size / 96)
    
    loads_M = np.reshape(loads, (n_days, 96))
    
    rpca = r_pca.R_pca(loads_M)
    L, S = rpca.fit(max_iter=10000, iter_print=1000)
    
    # visually inspect results (requires matplotlib)
    loads_filtered = L.flatten()
    dt = dict({'date-time' : timestamp, 'load' : loads_filtered})
    df = pd.DataFrame(dt)    
    df.to_csv('filtered/rpca_' + bld_name + '.csv', sep=',', index = False)

    
    
    