import glob
import outlier2
import data_cleaner
import pandas as pd


if __name__ == "__main__":
    file_list = []
    for file in glob.glob("*.csv"):
        file_list.append(str(file))
        
    for f in file_list:
        print('processing file:' + f)
        bld_name = f[:-4]
        
        start_time = data_cleaner.get_StartTime(bld_name)    
        end_time = '2012-07-06T00:00:00'
        
        # read data, make sure the time sequence is correct
        data_cleaner.read_rawData(bld_name, start_time, end_time)
        
        # read cleaned data, apply Kernel filter
        data = pd.read_csv('cleaned/' + bld_name + '_cleaned.csv')
        timestamp = data['timestamp'].values
        
        # call Kernel filter data cleansing function
        load_cleaned = outlier2.Kernel_clean(bld_name)
        d = dict({'date-time' : timestamp, 'load' : load_cleaned})
        df = pd.DataFrame(d)    
        df.to_csv('filtered/' + bld_name + '.csv', sep=',', index = False)
        