import numpy as np
import pandas as pd
from sklearn import preprocessing

def normalize_std(x):

    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    return scaler.transform(x)
    
def store_normed(filepath):
    
	data = filepath 
	data_strip_label = normalize_std(data.iloc[:,1:].values)
	data.iloc[:,1:] = data_strip_label
	
	del data_strip_label
	
	data = data.round(5) 
	return data.values

