import glob
import parse_file
import data_cleaner
import outlier_multi
import matplotlib.pyplot as plt
import pandas as pd


def getBldMap():
    bld_map = dict()
    
    for file in glob.glob('energy\*.csv'):
        bld_name = str(file[7:-4])
        bld_no = bld_name[0:4]
        if bld_no in bld_map.keys():
            bld_map[bld_no].append(bld_name)
        else:
            bld_map[bld_no] = [bld_name]
    return bld_map
    

def pltUtil(bld_name):
    print('plot building:' + bld_name)
    data = pd.read_csv('filtered/' + bld_name + '.csv')
    loads = data['load'].values
    plt.figure(figsize=(18,10))
    plt.plot(loads, 'b')
    plt.show()
    
    
if __name__ == "__main__":
    bld_map = getBldMap()
    
    figure_on = False
    
    for bld_names in bld_map.values():
        # single meter
        if len(bld_names) == 1:
            bld_name = bld_names[0]
            print('processing building:' + bld_name)
            print('parsing file...')
            parse_file.parsefile(bld_name)
            
            print('Creating load profile...')
            data_cleaner.GenkWLoad(bld_name, figure_on)
        
            print('Filtering abnormal data...')
            outlier_multi.Kernel_clean(bld_names, figure_on)
            
            pltUtil(bld_name)
        else:
            # multi meter
            for bld_name in bld_names:
                print('processing building:' + bld_name)
                print('parsing file...')
                parse_file.parsefile(bld_name)
                
                print('Creating load profile...')
                data_cleaner.GenkWLoad(bld_name, figure_on)
                
            print('Filtering abnormal data...')
            outlier_multi.Kernel_clean(bld_names, figure_on)
            
            pltUtil(bld_names[0])
    
