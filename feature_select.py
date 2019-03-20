import pandas as pd
import numpy as np


Wednesday_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/pro-Wednesday-0.5v2.csv'
Wednesday = pd.read_csv(Wednesday_path ,sep=",")

Monday_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/pro-Monday-0.5v2.csv'
Monday = pd.read_csv(Monday_path ,sep=",")

Tuesday_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/pro-Tuesday-0.5v2.csv'
Tuesday = pd.read_csv(Tuesday_path ,sep=",")

Thursday_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/pro-Thursday-0.5v2.csv'
Thursday = pd.read_csv(Thursday_path ,sep=",")

Friday_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/pro-Friday-0.5v2.csv'
Friday = pd.read_csv(Friday_path ,sep=",")


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
				amt_drop = int(0.95 * counter) # drop index of last 90%  of attack
				print( "initial amount: ", counter)
				print( "amt_drop: " , amt_drop)
				drop_idx = drop_idx[-amt_drop:] # indexes list of one paticular attack to drop 
				drop_idx_master += drop_idx
				drop_idx = []

	data_reduced = data.drop(drop_idx_master)
	print( "drop completed!!")
	return data_reduced


def clean_data(monday):
    
	monday_label = monday[["Label"]].copy()

	monday_trimed = monday.iloc[:, : len(monday.columns) -1 ].copy() #exclude the 'Label' columns

	monday_trimed = monday_trimed.apply(pd.to_numeric,errors='coerce')
	monday_trimed = monday_trimed.fillna(monday_trimed.mean())
	monday_trimed = monday_trimed.fillna(0.0)
	monday_label.loc[monday_label['Label'] == 'BENIGN' , "num_label"] = 0.0 #create new col "num_label" for numeric label
	monday_label.loc[monday_label['Label'] != 'BENIGN' , "num_label"] = 1.0

	monday_final = pd.concat([monday_label['num_label'], monday_trimed], axis = 1)
	monday_final = monday_final.round(5)

	return monday_final

Tuesday_reduced = drop_attacks(Tuesday)
Wednesday_reduced = drop_attacks(Wednesday)
Thursday_reduced = drop_attacks(Thursday)
Friday_reduced = drop_attacks(Friday)

all_train_reduced = pd.concat([Monday, Tuesday_reduced, Wednesday_reduced], axis = 0)
all_valid_reduced = pd.concat([Thursday_reduced], axis = 0)
all_test_reduced = pd.concat([Friday_reduced], axis = 0)

features_select = ['num_label', 'NumPkts', 'NumBytes', 'PktSizeAvg', 'PktSizeStd', 'NumACK', 'ULNumPkts', 'ULNumBytes', 'ULNumACK', 'DLNumPkts', 'DLNumBytes', 'DLPktSizeAvg', 'DLPktSizeStd', 'DLNumUniIPAddr', 'DLNumACK', 'DLPktSizeP25', 'PktSizeP75', 'DLPktSizeP75', 'ULnumActiveFlows', 'DLnumActiveFlows']

alltrainreduced_select_new = clean_data(all_train_reduced)[features_select]
allvalidreduced_select_new = clean_data(all_valid_reduced)[features_select]
alltestreduced_select_new = clean_data(all_test_reduced)[features_select]

alltrainreduced_select_new.to_csv('allreducedselect_w10o0_train.csv', 
                             sep=',', index = False, header = False)
alltestreduced_select_new.to_csv('allreducedselect_w10o0_test.csv', 
                             sep=',', index = False, header = False)
allvalidreduced_select_new.to_csv('allreducedselect_w10o0_validation.csv', 
                             sep=',', index = False, header = False)
