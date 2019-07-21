import pandas as pd
import numpy as np
from slidewindow import *
from shuffle_data import *
from normalization import *
import pdb

def drop_attacks(data):

	drop_idx_master = []
	drop_idx = []
	for i in range(len(data)):
		if data.iloc[i]["Label"] == 'BENIGN':
			counter = 0
		elif data.iloc[i]["Label"] != 'BENIGN':
			counter += 1
			drop_idx.append(i)
			if data.iloc[i+1]["Label"] == 'BENIGN':
				amt_drop = int(0.95 * counter) # drop index of last 95%  of attack
				print ("{} initial amount: {}".format(data.iloc[i]["Label"], counter))
				print ("amt_dropped: " , amt_drop)
				drop_idx = drop_idx[-amt_drop:] # indexes list of one paticular attack to drop 
				drop_idx_master += drop_idx
				drop_idx = []

	data_reduced = data.drop(drop_idx_master)
	print ("drop completed!!")
	return data_reduced


def clean_data(data):
	
	data_label = data[["Label"]].copy()
	
	data_trimed = data.iloc[:, : len(data.columns) -1 ].copy() #exclude the 'Label' columns
	
	data_trimed = data_trimed.apply(pd.to_numeric,errors='coerce')
	data_trimed = data_trimed.fillna(data_trimed.mean())
	data_trimed = data_trimed.fillna(0.0)
	data_label.loc[data_label['Label'] == 'BENIGN' , "num_label"] = 0.0 #create new col "num_label" for numeric label
	data_label.loc[data_label['Label'] != 'BENIGN' , "num_label"] = 1.0
	
	data_final = pd.concat([data_label['num_label'], data_trimed], axis = 1)
	data_final = data_final.round(5)
	
	return data_final


if __name__ == '__main__':

    # load data from csv files
	Monday_path = '../data/pro-Monday-0.5v2.csv'
	Monday = pd.read_csv(Monday_path ,sep=",")
	Tuesday_path = '../data/pro-Tuesday-0.5v2.csv'
	Tuesday = pd.read_csv(Tuesday_path ,sep=",")
	Wednesday_path = '../data/pro-Wednesday-0.5v2.csv'
	Wednesday = pd.read_csv(Wednesday_path ,sep=",")
	Thursday_path = '../data/pro-Thursday-0.5v2.csv'
	Thursday = pd.read_csv(Thursday_path ,sep=",")
	Friday_path = '../data/pro-Friday-0.5v2.csv'
	Friday = pd.read_csv(Friday_path ,sep=",")

	# create dataset that contains only 1% attack samples
	Tuesday_reduced = drop_attacks(Tuesday)
	Wednesday_reduced = drop_attacks(Wednesday)
	Thursday_reduced = drop_attacks(Thursday)
	Friday_reduced = drop_attacks(Friday)

	all_reduced = pd.concat([Monday, Tuesday_reduced, Wednesday_reduced, Thursday_reduced, Friday_reduced], axis = 0)

	# selected features from pcc score. refer to features_select_pcc.ipynb for more details 
	features_select = ['num_label', 'NumPkts', 'NumBytes', 'PktSizeAvg', 'PktSizeStd', 'NumACK', 'ULNumPkts', 'ULNumBytes', 
	'ULNumACK', 'DLNumPkts', 'DLNumBytes', 'DLPktSizeAvg', 'DLPktSizeStd', 'DLNumUniIPAddr', 'DLNumACK', 'DLPktSizeP25', 
	'PktSizeP75', 'DLPktSizeP75', 'ULnumActiveFlows', 'DLnumActiveFlows']

	allreduced_select_new = clean_data(all_reduced)[features_select]
	raw_labels =  all_reduced[['Label']].values
	
	#normalize data
	normed_all = store_normed(allreduced_select_new)

	#create sequence data with sliding window
	window_size = 10
	overlap = 9
	slide_all = create_windows(normed_all, window_size, overlap) 
	slide_string_labels = create_windows_raw_label(raw_labels, window_size, overlap) 

	print ('\n shuffling...')
	# shuffle sequence data and create train, val and test datasets
	Shuffle(slide_all).output_data()
	Shuffle(slide_string_labels).output_attack_idx()
