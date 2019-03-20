import numpy as np
import pandas as pd
import input_data_old
import pickle
import yaml


def att_percent(train_y):
	train_y = train_y.reshape(-1, 2)
	num_att = np.sum(train_y[:, 1]) 
	
	return float(num_att)/train_y.shape[0]


input_path = "normed_allreducedselect_w10o9_*" 
#input_path_raw_label_train = "/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/featuresv2/normed_allreducedselect_labels_w10o9_train.yml" 
#input_path_raw_label_valid = "/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/featuresv2/normed_allreducedselect_labels_w10o9_validation.yml" 
#input_path_raw_label_test = "/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/featuresv2/normed_allreducedselect_labels_w10o9_test.yml" 

train_data = input_data_old.Fixed_seq(input_path, None, None, None)
trainX, trainY = train_data.get_input('Train', SlidingAttackLabel = False)
#trainX, trainY = train_data.get_pickle('train')
#train_label = yaml.load(open(input_path_raw_label_train, 'r').read())
#train_label = train_label['label']


valid_data = input_data_old.Fixed_seq(input_path, None, None, None)
validX, validY = valid_data.get_input('Validation', SlidingAttackLabel = False)
#validX, validY = valid_data.get_pickle('validation')
#valid_label = yaml.load(open(input_path_raw_label_valid, 'r').read())
#valid_label = valid_label['label']



test_data = input_data_old.Fixed_seq(input_path, None, None, None)
testX, testY = test_data.get_input('Test', SlidingAttackLabel = False)
#testX, testY = test_data.get_pickle('test')
#test_label = yaml.load(open(input_path_raw_label_test, 'r').read())
#test_label = test_label['label']


#print( "initial train: " + str(att_percent(trainY)))
#print( "initial valid: " + str(att_percent(validY)))
#print( "initial test: " + str(att_percent(testY)))

def shuffle_within(dataX,dataY):
	np.random.seed(110)
	idx = np.arange(dataX.shape[0])
	np.random.shuffle(idx)

	dataX = dataX[idx]
	dataY = dataY[idx]
	return dataX, dataY

#train_x, train_y = shuffle_within(trainX, trainY)
#train_x = train_x[:11751]
#train_y = train_y[:11751]
#
#valid_x, valid_y = shuffle_within(validX, validY)
#valid_x = valid_x[:5850]
#valid_y = valid_y[:5850]
#
#test_x, test_y = shuffle_within(testX, testY)
#test_x = test_x[:5851]
#test_y = test_y[:5851]

dataX = np.concatenate((trainX, validX, testX), axis = 0)
dataY = np.concatenate((trainY, validY, testY), axis = 0)
#data_label = np.concatenate((train_label, valid_label, test_label), axis = 0)
#dataX = np.concatenate((validX, testX), axis = 0)
#dataY = np.concatenate((validY, testY), axis = 0)

#220
#125
np.random.seed(220)
idx = np.arange(dataX.shape[0])
np.random.shuffle(idx)

dataX = dataX[idx]
dataY = dataY[idx]
#data_label = data_label[idx]

m = dataX.shape[0]
#m = 22168
#
train_x = dataX[:int(0.6*m)] 
train_y = dataY[:int(0.6*m)] 
#train_label = data_label[:int(0.6*m)] 
train_att = att_percent(train_y)

print( train_y.shape)
print( "train: " + str(train_att))

#valid_x = dataX[:int(0.5*m) ] 
#valid_y = dataY[:int(0.5*m) ] 
valid_x = dataX[int(0.6*m):int(0.6*m) + int(0.2*m)] 
valid_y = dataY[int(0.6*m):int(0.6*m) + int(0.2*m)] 
#valid_label = data_label[int(0.6*m):int(0.6*m) + int(0.2*m)] 
valid_att = att_percent(valid_y)

print( valid_y.shape)
print( "valid: " + str(valid_att))


#test_x = dataX[int(0.5*m):] 
#test_y = dataY[int(0.5*m):] 
test_x = dataX[int(0.6*m) + int(0.2*m):]
test_y = dataY[int(0.6*m) + int(0.2*m):] 
#test_label = data_label[int(0.6*m) + int(0.2*m):] 
test_att = att_percent(test_y)

print( test_y.shape)
print( "test: " + str(test_att))

train = {'train_x': train_x, 'train_y': train_y}
valid = {'validation_x': valid_x, 'validation_y': valid_y}
test = {'test_x': test_x, 'test_y': test_y}


#train_raw_label = {'label': train_label}
#valid_raw_label = {'label': valid_label}
#test_raw_label = {'label': test_label}


with open("normed_allreducedselect_final_w10o9_train.pkl" , 'wb') as f:
	pickle.dump(train, f)
with open("normed_allreducedselect_final_w10o9_validation.pkl" , 'wb') as f:
	pickle.dump(valid, f)
with open("normed_allreducedselect_final_w10o9_test.pkl" , 'wb') as f:
	pickle.dump(test, f)

#train_raw_label_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/featuresv2/normed_allreducedselect_final_labels_w10o9_train.yml'
#valid_raw_label_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/featuresv2/normed_allreducedselect_final_labels_w10o9_validation.yml'
#test_raw_label_path = '/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/featuresv2/normed_allreducedselect_final_labels_w10o9_test.yml'
#
#fp = yaml.dump(train_raw_label)
#open(train_raw_label_path, 'w').write(fp)
#fp = yaml.dump(valid_raw_label)
#open(valid_raw_label_path, 'w').write(fp)
#fp = yaml.dump(test_raw_label)
#open(test_raw_label_path, 'w').write(fp)
