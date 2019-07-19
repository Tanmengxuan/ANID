import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import pdb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import h5py
import glob
import re

def output_layer(outputs, num_hidden, num_classes):

	final_output = tf.layers.dense(
			inputs= outputs,
			units= num_classes,
			activation= tf.nn.softmax,
			name="output_layer")

	
	return final_output

def weighted_crossentropy(y_true, y_pred):
	output = y_pred
	#pdb.set_trace()
	output /= tf.reduce_sum(y_pred, -1, True)
	output = tf.clip_by_value(output, 1e-5, 1. - 1e-5)
	#xent = - tf.reduce_sum(y_true * tf.log(output), -1)
	# pos_weight = 5.87
	# xent *= y_true[:,:,1] * pos_weight + y_true[:,:,0] * 1
	#xent *=   y_true[:,:,1] * (tf.cast(tf.reduce_sum(y_true[:, :, 0]), tf.float32) / tf.reduce_sum(y_true[:, :, 1])) + y_true[:, :, 0] * 1 
	pos_weight = tf.cast(tf.reduce_sum(y_true[:, :, 0]), tf.float32) / (tf.reduce_sum(y_true[:, :, 1]) +1e-5)
	loss = y_true[:,:,1]*pos_weight*tf.log(output)[:,:,1] + y_true[:,:,0]*1*tf.log(output)[:,:,0]
	xent = - tf.reduce_mean(loss)
	return xent

def get_result(test_pred, testY, threshold):
	#pdb.set_trace()
	pred = test_pred.reshape(-1, test_pred.shape[2])
	pred = pred > threshold
	pred = pred.astype(int)
	
	targ = testY.reshape( -1, testY.shape[2])

	pred_attack = pred[:, 1]
	targ_attack = targ[:, 1]
	conf = confusion_matrix(targ_attack, pred_attack)
	
	_test_precision,_test_recall,_test_f1,_support = precision_recall_fscore_support(targ, pred)
	
	return _test_precision, _test_recall, _test_f1, _support, pred, targ, conf

def plot_performance(train_loss, val_loss, train_f1, val_f1, title):

	labels = [ "train_loss", "val_loss", "train_f1", "val_f1" ]
	
	plot_data = dict()
	plot_data["train_loss"] = train_loss
	plot_data["val_loss"] = val_loss
	plot_data["train_f1"] = train_f1
	plot_data["val_f1"] = val_f1
	
	
	fig, ax = plt.subplots(figsize = (10,8))
	for metric in labels:
		ax.plot( np.arange(len(plot_data[metric])) , np.array(plot_data[metric]), label=metric)
	ax.legend()
	ax.set_ylim(0.0, 1)
	plt.title(title)
	plt.xlabel('no. of epoch')
	fig.savefig('/home/tmx/cisdem/CrossFire-Detect/models/plot_train/' + title + '.png')
	#plt.show()

def plot_performance_crf(train_loss, val_loss, val_f1, title):

	labels = [ "train_loss", "val_loss", "val_f1" ]
	
	plot_data = dict()
	plot_data["train_loss"] = train_loss
	plot_data["val_loss"] = val_loss
	plot_data["val_f1"] = val_f1
	
	
	fig, ax = plt.subplots(figsize = (10,8))
	for metric in labels:
		ax.plot( np.arange(len(plot_data[metric])) , np.array(plot_data[metric]), label=metric)
	ax.legend()
	#ax.set_ylim(0.0, 1)
	plt.title(title)
	plt.xlabel('no. of epoch')
	fig.savefig('/home/tmx/cisdem/CrossFire-Detect/models/plot_train/' + title + '.png')
	#plt.show()

def shuffle_fixed_data(data, label):
    
	idx = np.arange(len(data)) #create array of index for data
	np.random.shuffle(idx) #randomly shuffle idx
	data = data[idx]
	label = label[idx]
	
	return data, label

def get_total_para(trainable_variables):

	total_parameters = 0
	for variable in trainable_variables:
		shape = variable.get_shape()
		variable_parameters = 1

		for dim in shape:
			variable_parameters *= dim.value

		total_parameters += variable_parameters
	print('total_parameters: ', total_parameters)

class Inputter(object):

	def __init__(self, filepathGlob):
		
		files = glob.glob(filepathGlob)

		for dataFile in files:
			if re.search(r'test\.*', dataFile):
				self.testFile = dataFile
				print ('test file read: ', dataFile)

			elif re.search(r'train\.*', dataFile):
				self.trainFile = dataFile
				print ('train file read: ', dataFile)

			elif re.search(r'valid\.*', dataFile):
				self.validFile = dataFile
				print ('valid file read: ', dataFile)

	def load_data(self, name):

		file_path = getattr(self, name + 'File')		
		inputX = h5py.File(file_path, 'r').get('data')[:]
		inputY = h5py.File(file_path, 'r').get('label')[:]

		print ('\n {} data shape: {} {}'.format(name, inputX.shape, inputY.shape))

		return inputX, inputY

class Create_batch(object):
	
	def __init__(self, data, label, batch_size):

		self.data = data
		self.label = label
		self.batch_size = batch_size
		self.start_index = 0

	def get_num_batch(self):

		num_batch = int(len(self.data)/self.batch_size) + min( len(self.data)%self.batch_size, 1)
		return num_batch

	def nextbatch(self):

		try:
			batch_x = self.data[self.start_index : self.start_index + self.batch_size]
			batch_y = self.label[self.start_index : self.start_index + self.batch_size]
		except:
			batch_x = self.data[self.start_index: ]
			batch_y = self.label[self.start_index: ]

		self.start_index += self.batch_size

		return batch_x, batch_y

class Attacks(object):

	def __init__(self, attack_type):
		
		self.attack_type = attack_type
		#self.attack_idx = h5py.File('h5files/test_allattack_idx.h5', 'r').get(self.attack_type)[:]
		self.attack_idx = h5py.File('/home/tmx/cicids_alfonso/test_new_allattack_idx.h5', 'r').get(self.attack_type)[:]
		self.threshold = 0.5
		self.tp = 0
		self.fp = 0
		self.tn = 0
		self.fn = 0
		self.N_attacks = 6049.0 # total number of attack samples in testset

	def compute_all_attacks(self, fp_all, test_pred, testY):

		_, _, _, tp, fp, tn, fn = self.compute_fnr(fp_all, self.attack_idx, test_pred, testY, self.threshold)

		self.tp += tp
		self.fp += fp
		self.tn += tn
		self.fn += fn


	def compute_fnr(self, fp_all, idx_list, test_pred, testY, threshold):
	
		#pdb.set_trace()
		test_pred_attack = test_pred[idx_list, :, :]
		testY_attack = testY[idx_list, :, :]

		_test_precision,_test_recall,_test_f1,_support, pred, targ, conf = get_result(test_pred_attack, testY_attack, threshold)
		
		recall = _test_recall[1]

		tp = conf[1][1]
		n_i = conf[1][0] + conf[1][1] # total attack_i samples = fn + tp
		precision,fp = self.compute_prec_attack( tp, n_i, self.N_attacks, fp_all)  

		f1 = (2 * recall * precision) / (recall + precision) 

		tn = conf[0][0]
		fn = conf[1][0]
		#return precision, recall, f1
		return precision, recall, f1, tp, fp, tn, fn 

	def compute_prec_attack(self, tp, n_i, N_attacks, fp_all):

		fp = fp_all * (n_i/N_attacks) # fp is proportional to the % of that attack
		prec = tp /( tp + fp)

		#return prec
		return prec,fp


	def get_result(self):

		prec = self.tp / (self.tp + self.fp)
		rec = self.tp/(self.tp + self.fn)
		f1 = 2*prec*rec / (prec + rec)
		
		print ( '\n Attack: {}, f1: {:.4f}, prec: {:.4f}, rec: {:.4f}'.format(self.attack_type, f1, prec, rec))

class Multi(object):

	def __init__(self):

		self.tp = 0
		self.fp = 0
		self.tn = 0
		self.fn = 0

	def compute(self, conf):

		self.tp += conf[1][1]
		self.fp += conf[0][1]
		self.tn += conf[0][0] 
		self.fn += conf[1][0]

	def get_result(self):

		prec = self.tp / (self.tp + self.fp)
		rec = self.tp/(self.tp + self.fn)
		f1 = 2*prec*rec / (prec + rec)
		fpr = self.fp / (self.fp + self.tn)
		
		print ( '\n Final f1: {:.4f}, prec: {:.4f}, rec: {:.4f}, fpr: {:.4f}'.format(f1, prec, rec, fpr))




