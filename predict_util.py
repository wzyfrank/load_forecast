import numpy as np

def calMAPE(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calRMSPE(y_true, y_pred): 
    return np.sqrt((((y_true - y_pred)/y_true) ** 2).mean()) * 100