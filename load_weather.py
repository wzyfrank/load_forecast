import pandas as pd
import datetime
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import time


def get_StartTime(bld_name):
    loads_list = pd.read_csv(bld_name + '.csv', sep = ',')
    first_time = loads_list.iloc[0, 0]
    return first_time


def ReadWeather():
    weather_year = 'weather/seattle'
    weather_list = pd.read_csv(weather_year + '.csv', sep = ',')
    N_lines = len(weather_list.index)
    
    DewPoint = np.zeros((N_lines))
    DHI = np.zeros((N_lines))
    DNI = np.zeros((N_lines))
    Temp = np.zeros((N_lines))
    Pressure = np.zeros((N_lines))
    Humidity = np.zeros((N_lines))
    WindSpeed = np.zeros((N_lines))
    
    times = np.zeros((N_lines))
    timestamp = []
    for i in range(N_lines):
        year = int(weather_list.iloc[i, 0])
        month = int(weather_list.iloc[i, 1])
        day = int(weather_list.iloc[i, 2])
        hour = int(weather_list.iloc[i, 3])
        minute = int(weather_list.iloc[i, 4])
        
        d = datetime.datetime(year, month, day, hour, minute, 0)
        timestamp.append(d)
        times[i] = int(time.mktime(d.timetuple()))
        
        DewPoint[i] = float(weather_list.iloc[i, 5])
        DHI[i] = float(weather_list.iloc[i, 6])
        DNI[i] = float(weather_list.iloc[i, 7])
        Temp[i] = float(weather_list.iloc[i, 8])
        Pressure[i] = float(weather_list.iloc[i, 9])
        Humidity[i] = float(weather_list.iloc[i, 10])
        WindSpeed[i] = float(weather_list.iloc[i, 11])
        
    return(times, DewPoint, DHI, DNI, Temp, Pressure, Humidity, WindSpeed)
    
    
def Interpolate(bld_name, times, DewPoint, DHI, DNI, Temp, Pressure, Humidity, WindSpeed):
    fTemp = interpolate.interp1d(times, Temp)
    fDewPoint = interpolate.interp1d(times, DewPoint)
    fDHI = interpolate.interp1d(times, DHI)
    fDNI = interpolate.interp1d(times, DNI)
    fPressure = interpolate.interp1d(times, Pressure)
    fHumidity = interpolate.interp1d(times, Humidity)
    fWindSpeed = interpolate.interp1d(times, WindSpeed)
    
    start_time = get_StartTime(bld_name)    
    end_time = '2012-07-06 00:00:00'
    start_t = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    start_t = int(time.mktime(start_t.timetuple()))
    end_t = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    end_t = int(time.mktime(end_t.timetuple()))
    
    vlen = int((end_t - start_t) / 900)
    new_time = np.linspace(start_t, end_t - 900, vlen)
    
    nTemp = fTemp(new_time)
    nDewPoint = fDewPoint(new_time)
    nDHI = fDHI(new_time)
    nDNI = fDNI(new_time)
    nPressure = fPressure(new_time)
    nHumidity = fHumidity(new_time)
    nWindSpeed = fWindSpeed(new_time)
    
    return(nTemp, nDewPoint, nDHI, nDNI, nPressure, nHumidity, nWindSpeed)
    

def CalCorrelation(bld_name, nTemp, nDewPoint, nDHI, nDNI, nPressure, nHumidity, nWindSpeed):
    data = pd.read_csv(bld_name + '.csv')
    timestamp = data['date-time'].values
    load = data['load'].values
    
    load_len = load.size
    load_weekday = []
    Temp_weekday = []
    DewPoint_weekday = []
    DHI_weekday = []
    DNI_weekday = []
    Pressure_weekday = []
    Huminity_weekday = []
    WindSpeed_weekday = []
    
    t = 0
    while t < load_len:
        d = datetime.datetime.strptime(timestamp[t], '%Y-%m-%d %H:%M:%S')
        # check whether the date is weekday
        if d.isoweekday() in range(1, 6):
            load_weekday.extend(load[t:t+96])
            Temp_weekday.extend(nTemp[t:t+96])
            DewPoint_weekday.extend(nDewPoint[t:t+96])
            DHI_weekday.extend(nDHI[t:t+96])
            DNI_weekday.extend(nDNI[t:t+96])
            Pressure_weekday.extend(nPressure[t:t+96])
            Huminity_weekday.extend(nHumidity[t:t+96])
            WindSpeed_weekday.extend(nWindSpeed[t:t+96])
        t += 96
    
    load_weekday = np.asarray(load_weekday)
    Temp_weekday = np.asarray(Temp_weekday)
    DewPoint_weekday = np.asarray(DewPoint_weekday)
    DHI_weekday = np.asarray(DHI_weekday)
    DNI_weekday = np.asarray(DNI_weekday)
    Pressure_weekday = np.asarray(Pressure_weekday)
    Huminity_weekday = np.asarray(Huminity_weekday)
    WindSpeed_weekday = np.asarray(WindSpeed_weekday)
    
    Temp_coef = np.corrcoef(load_weekday, Temp_weekday)[0,1]
    DewPoint_coef = np.corrcoef(load_weekday, DewPoint_weekday)[0, 1]
    DHI_coef = np.corrcoef(load_weekday, DHI_weekday)[0,1]
    DNI_coef = np.corrcoef(load_weekday, DNI_weekday)[0,1]
    Pressure_coef = np.corrcoef(load_weekday, Pressure_weekday)[0,1]
    Huminity_coef = np.corrcoef(load_weekday, Huminity_weekday)[0,1]
    WindSpeed_coef = np.corrcoef(load_weekday, WindSpeed_weekday)[0,1]
    
    # 
    weekday_len = load_weekday.size
    day_period = np.zeros((weekday_len))
    week_period = np.zeros((weekday_len))
    Td = 96
    Tw = 96 * 5
    for i in range(weekday_len):
        day_period[i] = np.cos(2 * np.pi * i / Td)
        week_period[i] = np.cos(2 * np.pi * i / Tw)
    
    Day_coef = np.corrcoef(load_weekday, day_period)[0,1]
    Week_coef = np.corrcoef(load_weekday, week_period)[0,1]
    
    return(Temp_coef, DewPoint_coef, DHI_coef, DNI_coef, Pressure_coef, Huminity_coef, WindSpeed_coef, Day_coef, Week_coef)
    

def getWeatherData(bld_name):
    (times, DewPoint, DHI, DNI, Temp, Pressure, Humidity, WindSpeed) = ReadWeather()
    (nTemp, nDewPoint, nDHI, nDNI, nPressure, nHumidity, nWindSpeed) = Interpolate(bld_name, times, DewPoint, DHI, DNI, Temp, Pressure, Humidity, WindSpeed)

    data = pd.read_csv(bld_name + '.csv')
    timestamp = data['date-time'].values
    load = data['load'].values    
    load_len = load.size
    
    Temp_weekday = []
    Huminity_weekday = []
    WindSpeed_weekday = []      

    t = 0
    while t < load_len:
        d = datetime.datetime.strptime(timestamp[t], '%Y-%m-%d %H:%M:%S')
        # check whether the date is weekday
        if d.isoweekday() in range(1, 6):
            Temp_weekday.extend(nTemp[t:t+96])
            Huminity_weekday.extend(nHumidity[t:t+96])
            WindSpeed_weekday.extend(nWindSpeed[t:t+96])
        t += 96
    
    Temp_weekday = np.asarray(Temp_weekday)
    Huminity_weekday = np.asarray(Huminity_weekday)
    WindSpeed_weekday = np.asarray(WindSpeed_weekday)
    
    weekday_len = Temp_weekday.size
    Day_period = np.zeros((weekday_len))
    Td = 96
    for i in range(weekday_len):
        Day_period[i] = np.cos(2 * np.pi * i / Td)
    
    return(Temp_weekday, Huminity_weekday, WindSpeed_weekday, Day_period)   
          
if __name__ == "__main__":

    bld_name = 'combined/1008_EE_CSE'
    # building start and end times
    start_time = get_StartTime(bld_name)  
    end_time = '2012-07-06 00:00:00'
    
    '''    
    # read weather data
    (times, DewPoint, DHI, DNI, Temp, Pressure, Humidity, WindSpeed) = ReadWeather()
    
    # interpolate
    (nTemp, nDewPoint, nDHI, nDNI, nPressure, nHumidity, nWindSpeed) = Interpolate(times, DewPoint, DHI, DNI, Temp, Pressure, Humidity, WindSpeed)
    
    # calculate the correlation
    (Temp_coef, DewPoint_coef, DHI_coef, DNI_coef, Pressure_coef, Huminity_coef, WindSpeed_coef, Day_coef, Week_coef) = CalCorrelation(bld_name, nTemp, nDewPoint, nDHI, nDNI, nPressure, nHumidity, nWindSpeed)
    print(Temp_coef, DewPoint_coef, DHI_coef, DNI_coef, Pressure_coef, Huminity_coef, WindSpeed_coef, Day_coef, Week_coef)
    '''
    
    (Temp_weekday, Huminity_weekday, WindSpeed_weekday, Day_period) = getWeatherData(bld_name)