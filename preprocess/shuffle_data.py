import numpy as np
import h5py
import pdb

class Shuffle(object):

	def __init__(self, data):
		
		self.data = data 
	
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

		save_path = '../data/' + name + '_allattack_idx.h5'
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
		save_path = '../data/normed_w10o9_{}.h5'.format(name)
		hf = h5py.File(save_path, 'w')
		hf.create_dataset('data', data = dataX)
		hf.create_dataset('label', data = dataY)
		hf.close()

		print ('file saved: ', save_path)
		print ('{} att %: {:.3f}'.format(name, self.att_percent(dataY)))

	def output_attack_idx(self):

		train, valid, test = self.shuffle(self.data)
		self.save_attack_idx(train, 'train')
		self.save_attack_idx(valid, 'validation')
		self.save_attack_idx(test, 'test')

	def output_data(self):

		train, valid, test = self.shuffle(self.data)
		self.save_data(train, 'train')
		self.save_data(valid, 'validation')
		self.save_data(test, 'test')
