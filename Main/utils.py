import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import pdb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import h5py
import glob
import re
from datetime import datetime

def weighted_crossentropy(y_true, y_pred):

	output = y_pred
	output /= tf.reduce_sum(y_pred, -1, True)
	output = tf.clip_by_value(output, 1e-5, 1. - 1e-5)
	pos_weight = tf.cast(tf.reduce_sum(y_true[:, :, 0]), tf.float32) / (tf.reduce_sum(y_true[:, :, 1]) +1e-5)
	loss = y_true[:,:,1]*pos_weight*tf.log(output)[:,:,1] + y_true[:,:,0]*1*tf.log(output)[:,:,0]
	xent = - tf.reduce_mean(loss)
	return xent

def evaluate_test(testX, testY,Input_X, Input_Y, keep_prob, sess, loss, final_output, batch_size):

	'''function used to do inference on validation and test datasets'''

	get_test_batch = Create_batch(testX, testY, batch_size)
	num_test_batch = get_test_batch.get_num_batch()

	#pdb.set_trace()
	test_batch_loss = 0
	for batch in range(num_test_batch):
		mini_test_x, mini_test_y = get_test_batch.nextbatch()

		test_loss, test_pred = sess.run([loss, final_output], 
							feed_dict={Input_X: mini_test_x,
							Input_Y: mini_test_y,
							keep_prob: 1.0,
							})
		test_batch_loss += test_loss

		if batch == 0:
			test_pred_init = test_pred
		else:
			test_pred_init = np.concatenate([test_pred_init, test_pred], axis =0)

	#pdb.set_trace()	
	test_batch_loss = test_batch_loss/num_test_batch		

	return test_batch_loss, test_pred_init

def get_result(test_pred, testY, threshold):

	'''function used to compute F1 scores''' 

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
	fig.savefig('../plot_train/' + title + '.png')
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
	fig.savefig('../plot_train/' + title + '.png')
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

def test_sample(sess, testX, Input_X, keep_prob, final_output):

	'''function used to compute inference time of a single sample'''

	one_sample = testX[[0], :, :]
	past = datetime.now()
	prediction = sess.run([final_output],
						feed_dict = {Input_X: one_sample,
									keep_prob:1.0})
	print ("test time one sample {:.5f}s".format(( datetime.now() - past).total_seconds())) 

class Inputter(object):

	def __init__(self, filepathGlob):
		
		#pdb.set_trace()
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
		self.attack_idx = h5py.File('../data/test_allattack_idx.h5', 'r').get(self.attack_type)[:]
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
