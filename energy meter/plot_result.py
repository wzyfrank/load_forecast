import glob
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    
    bld_names = []
    for file in glob.glob('filtered\*.csv'):
        bld_names.append(str(file[9:-4]))   
    
    for bld_name in bld_names:
        print('plot building:' + bld_name)
        data = pd.read_csv('filtered/' + bld_name + '.csv')
        loads = data['load'].values
        timestamp = data['date-time'].values
        plt.figure(figsize=(18,10))
        plt.plot(loads, 'b')
        plt.show()
