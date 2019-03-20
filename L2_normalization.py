import numpy as np
import pandas as pd
from sklearn import preprocessing
import glob
import re
import argparse
import time

def normalize_std(x):
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    return scaler.transform(x)
    
def store_normed(filepath):
    
	start_time = time.time()
	
	#data = pd.read_csv(filepath,sep="\t", index_col=None, header=None, dtype='float64')
	data = pd.read_csv(filepath,sep=",", index_col=None, header=None, dtype='float64')
	data_strip_label = normalize_std(data.iloc[:,1:].values)
	data.iloc[:,1:] = data_strip_label
	
	del data_strip_label
	
	newpath = re.sub(r'(\S+\w10o0\D+)', r'normed_\1', filepath)
	
	print ("normed file stored at {}: ".format(newpath))
	
	data = data.round(5) 
	data.to_csv(newpath ,index=False,sep="\t",header=False)
	
	print("--- %s minutess ---" % ((time.time() - start_time)/60.0))
	print ("\n")

    
parser = argparse.ArgumentParser()
parser.add_argument("--path",help="Input the file path")
args = parser.parse_args()
    

if args.path:
    path = args.path
    
files = glob.glob(path)    
    
for filename in files:
    if re.search(r'test\.csv',filename):
        testfile_path = filename
        print ("test file read: "+ testfile_path)
        print ("\n")
        store_normed(testfile_path)
        
    if re.search(r'validation\.csv',filename):
        validationfile_path = filename
        print ("validation file read: "+ validationfile_path)
        print ("\n")
        store_normed(validationfile_path)
        
    if re.search(r'train\.csv',filename):
        trainfile_path = filename
        print ("train file read: "+ trainfile_path)
        print ("\n")
        store_normed(trainfile_path)
