import numpy as np
import pandas as pd
#import input_data_old
import pickle
import yaml
import h5py
import pdb



def shuffle_within(dataX,dataY):
	np.random.seed(110)
	idx = np.arange(dataX.shape[0])
	np.random.shuffle(idx)

	dataX = dataX[idx]
	dataY = dataY[idx]
	return dataX, dataY

def get_attack_idx (attack_type_list, test_label_raw):

	idx_list = []
	for idx in range(len(test_label_raw)):
		for attack in attack_type_list:
			if test_label_raw[idx][0] == attack  or test_label_raw[idx][-1] == attack :
				idx_list.append(idx)

	return idx_list
################################################for pure data#####################################
#input_path = "normed_allreducedselect_w10o9_*" 
#
#train_data = input_data_old.Fixed_seq(input_path, None, None, None)
#trainX, trainY = train_data.get_input('Train', SlidingAttackLabel = False)
#
#valid_data = input_data_old.Fixed_seq(input_path, None, None, None)
#validX, validY = valid_data.get_input('Validation', SlidingAttackLabel = False)
#
#test_data = input_data_old.Fixed_seq(input_path, None, None, None)
#testX, testY = test_data.get_input('Test', SlidingAttackLabel = False)
#
#dataX = np.concatenate((trainX, validX, testX), axis = 0)
#dataY = np.concatenate((trainY, validY, testY), axis = 0)
#
#np.random.seed(220)
#idx = np.arange(dataX.shape[0])
#np.random.shuffle(idx)
#
#dataX = dataX[idx]
#dataY = dataY[idx]
#
#m = dataX.shape[0]
#
#train_x = dataX[:int(0.6*m)] 
#train_y = dataY[:int(0.6*m)] 
#train_att = att_percent(train_y)
#print( train_y.shape)
#print( "train: " + str(train_att))
#
#valid_x = dataX[int(0.6*m):int(0.6*m) + int(0.2*m)] 
#valid_y = dataY[int(0.6*m):int(0.6*m) + int(0.2*m)] 
#valid_att = att_percent(valid_y)
#print( valid_y.shape)
#print( "valid: " + str(valid_att))
#
#test_x = dataX[int(0.6*m) + int(0.2*m):]
#test_y = dataY[int(0.6*m) + int(0.2*m):] 
#test_att = att_percent(test_y)
#print( test_y.shape)
#print( "test: " + str(test_att))
#
#train = {'train_x': train_x, 'train_y': train_y}
#valid = {'validation_x': valid_x, 'validation_y': valid_y}
#test = {'test_x': test_x, 'test_y': test_y}
#
#
#with open("normed_allreducedselect_final_w10o9_train.pkl" , 'wb') as f:
#	pickle.dump(train, f)
#with open("normed_allreducedselect_final_w10o9_validation.pkl" , 'wb') as f:
#	pickle.dump(valid, f)
#with open("normed_allreducedselect_final_w10o9_test.pkl" , 'wb') as f:
#	pickle.dump(test, f)



################################################for string label#####################################


#input_path_raw_label_train = "normed_allreducedselect_labels_w10o9_train.yml" 
#input_path_raw_label_valid = "normed_allreducedselect_labels_w10o9_validation.yml" 
#input_path_raw_label_test = "normed_allreducedselect_labels_w10o9_test.yml" 
#
#train_label = yaml.load(open(input_path_raw_label_train, 'r').read())
#train_label = train_label['label']
#
#valid_label = yaml.load(open(input_path_raw_label_valid, 'r').read())
#valid_label = valid_label['label']
#
#test_label = yaml.load(open(input_path_raw_label_test, 'r').read())
#test_label = test_label['label']
#
#data_label = np.concatenate((train_label, valid_label, test_label), axis = 0)
#
#np.random.seed(220)
#idx = np.arange(data_label.shape[0])
#np.random.shuffle(idx)
#
#data_label = data_label[idx]
#
#m = data_label.shape[0]
#
#train_label = data_label[:int(0.6*m)] 
#valid_label = data_label[int(0.6*m):int(0.6*m) + int(0.2*m)] 
#test_label = data_label[int(0.6*m) + int(0.2*m):] 
#
#train_raw_label = {'label': train_label}
#valid_raw_label = {'label': valid_label}
#test_raw_label = {'label': test_label}



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



#dos = [u'DoS Go', u'DoS sl', u'DoS Sl',u'DoS Hu']
#ddos = [u'DDoS LOIT']
#infl = [u'Infiltration \u2013 Cool disk \u2013 MAC', u'Infiltration \u2013 Dropbox download Win Vista',
#	    u'Infiltration \u2013 Dropbox download - Meta exploit Win Vista' ]
#brute = [u'Brute ']
#port = [u'Port Scan \u2013 Firewall Rule off',u'Port Scan \u2013 Firewall Rule on' ]
#web = [u'Heartb', u'Web Attack \u2013 XSS', u'Web Attack \u2013 Brute Force',u'Web Attack \u2013 Sql Injection']
#botnet = [u'Botnet ARES']
#benign = [u'BENIGN']
#
#
#dos_idx = get_attack_idx(dos, test_label)
#ddos_idx = get_attack_idx(ddos, test_label)
#infl_idx = get_attack_idx(infl, test_label)
#brute_idx = get_attack_idx(brute, test_label)
#port_idx = get_attack_idx(port, test_label)
#web_idx = get_attack_idx(web, test_label)
#botnet_idx = get_attack_idx(botnet, test_label)
#benign_idx = get_attack_idx(benign, test_label)
#
#hf = h5py.File('test_allattack_idx.h5', 'w')
#hf.create_dataset('dos', data = dos_idx)
#hf.create_dataset('ddos', data = ddos_idx)
#hf.create_dataset('infl', data = infl_idx)
#hf.create_dataset('brute', data = brute_idx)
#hf.create_dataset('port', data = port_idx)
#hf.create_dataset('web', data = web_idx)
#hf.create_dataset('botnet', data =botnet_idx)
#hf.create_dataset('benign', data =benign_idx)
#hf.close()


class Shuffle(object):

	def __init__(self, train, valid, test):
		
		self.train = train
		self.valid = valid
		self.test = test
	
	def shuffle(self, data):

		np.random.seed(220)
		idx = np.arange(data.shape[0])
		np.random.shuffle(idx)
		data = data[idx]

		m = data.shape[0]

		train = data[:int(0.6*m)] 
		valid = data[int(0.6*m):int(0.6*m) + int(0.2*m)] 
		test = data[int(0.6*m) + int(0.2*m):] 

		return train, valid, test
	
	def get_attack_idx (self, attack_type_list, test_label_raw):
		
		#pdb.set_trace()
		idx_list = []
		for idx in range(len(test_label_raw)):
			for attack in attack_type_list:
				if test_label_raw[idx][0] == attack  or test_label_raw[idx][-1] == attack :
					idx_list.append(idx)

		return idx_list
	
	def att_percent(self, trainY):

		trainY = trainY.reshape(-1, 2)
		num_att = np.sum(trainY[:, 1]) 
	
		return float(num_att)/trainY.shape[0]

	def onehot(self, dataY):

		att = dataY
		benign = 1 - att 

		return np.concatenate( (benign, att), axis = -1 )

	def save_attack_idx(self, shuffled_data, name):

		dos = ['DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk']
		ddos = ['DDoS LOIT']
		infl = ['Infiltration – Cool disk – MAC', 'Infiltration – Dropbox download Win Vista',
				'Infiltration – Dropbox download - Meta exploit Win Vista' ]
		brute = ['Brute Force - FTP-Patator', 'Brute Force - SSH-Patator']
		port = ['Port Scan – Firewall Rule off', 'Port Scan – Firewall Rule on' ]
		web = ['Heartbleed Port 444', 'Web Attack – XSS', 'Web Attack – Brute Force', 'Web Attack – Sql Injection']
		botnet = ['Botnet ARES']
		benign = ['BENIGN']

		dos_idx = self.get_attack_idx(dos, shuffled_data)
		ddos_idx = self.get_attack_idx(ddos, shuffled_data)
		infl_idx = self.get_attack_idx(infl, shuffled_data)
		brute_idx = self.get_attack_idx(brute, shuffled_data)
		port_idx = self.get_attack_idx(port, shuffled_data)
		web_idx = self.get_attack_idx(web, shuffled_data)
		botnet_idx = self.get_attack_idx(botnet, shuffled_data)
		benign_idx = self.get_attack_idx(benign, shuffled_data)

		save_path = name + '_allattack_idx.h5'
		hf = h5py.File(save_path, 'w')
		hf.create_dataset('dos', data = dos_idx)
		hf.create_dataset('ddos', data = ddos_idx)
		hf.create_dataset('infl', data = infl_idx)
		hf.create_dataset('brute', data = brute_idx)
		hf.create_dataset('port', data = port_idx)
		hf.create_dataset('web', data = web_idx)
		hf.create_dataset('botnet', data =botnet_idx)
		hf.create_dataset('benign', data =benign_idx)
		hf.close()

		print ('file saved: ', save_path)

	def save_data(self, data, name):

		dataY = data[:, :, [0]]
		dataY = self.onehot(dataY)

		dataX = data[:, :, 1:]
		#pdb.set_trace()
		save_path = 'normed_allreducedselect_final_w10o9_{}.h5'.format(name)
		hf = h5py.File(save_path, 'w')
		hf.create_dataset('data', data = dataX)
		hf.create_dataset('label', data = dataY)
		hf.close()

		print ('file saved: ', save_path)
		print ('{} att %: {:.3f}'.format(name, self.att_percent(dataY)))

	def output_attack_idx(self):

		data = np.concatenate((self.train, self.valid, self.test), axis = 0)

		train, valid, test = self.shuffle(data)
		self.save_attack_idx(train, 'train')
		self.save_attack_idx(valid, 'validation')
		self.save_attack_idx(test, 'test')

	def output_data(self):

		data = np.concatenate((self.train, self.valid, self.test), axis = 0)

		train, valid, test = self.shuffle(data)
		self.save_data(train, 'train')
		self.save_data(valid, 'validation')
		self.save_data(test, 'test')
