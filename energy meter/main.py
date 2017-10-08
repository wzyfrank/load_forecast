import glob
import parse_file
import data_cleaner
import outlier

if __name__ == "__main__":
    
    bld_names = []
    for file in glob.glob("energy/*.csv"):
        bld_names.append(str(file))   
    
    for bld_name in bld_names:
        print('processing building:' + bld_name)
        print('parsing file...')
        parse_file.parsefile(bld_name)
        
        print('Creating load profile...')
        figure_on = False
        start_time = data_cleaner.get_StartTime(bld_name)
        end_time = data_cleaner.get_EndTime(bld_name) 
        data_cleaner.GenkWLoad(bld_name, start_time, end_time, figure_on)
    
        print('Filtering abnormal data...')
        outlier.Kernel_clean(bld_name, figure_on)
        
        print()
