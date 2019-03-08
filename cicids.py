import matplotlib
matplotlib.use('Agg')
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

import input_data_old
from utils import *
from os import mkdir,environ
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from absl import flags
from absl import app

import pdb

import attention_main as att
import bilstm_main as bi

FLAGS = flags.FLAGS
#FLAGS(sys.argv)
# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("plot", False, "Visualise attention")
flags.DEFINE_integer("sample_idx", 0, "Sample idx in testset to Visualise")

# Models parameters
flags.DEFINE_bool("atten", False, "Apply attention")
flags.DEFINE_float("drop", 0.7, "lstm dropout probability")
flags.DEFINE_integer("num_hidden", 600, "lstm size")
flags.DEFINE_integer("num_features", 193, "feature size of input")
flags.DEFINE_integer("MAXLEN", 10, "window size of lstm")

# Training parameters
flags.DEFINE_integer("num_epoch", 50, "number of training epochs")
flags.DEFINE_integer("batch_size", 256, "batch size of training datasets")

# Paths
flags.DEFINE_string("model_name", "cicids", "name of saved model")
flags.DEFINE_string("save_path", "/home/tmx/cisdem/CrossFire-Detect/models/lstm_runs/cicids/wedattack/new_features/", "path of saved model")
flags.DEFINE_string("input_path",
"/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/normed_allreducedselect_final_w10o9*",
"input path of data") 

def get_result(test_pred, testY, threshold):
	pred = test_pred.reshape(-1, test_pred.shape[2])
	pred = np.where(np.isnan(pred), 0, pred)
	pred = np.clip(pred, 1e-5, 1. - 1e-5) 
	pred = pred > threshold
	pred = pred.astype(int)
	pred_pd = pd.DataFrame(pred)
	pred_pd.rename(columns = {0:"pred_nonattack", 1:"pred_attack"}, inplace = True)
	
	targ = testY.reshape( -1, testY.shape[2])
	targ_pd = pd.DataFrame(targ)
	targ_pd.rename(columns = {0:"targ_nonattack", 1:"targ_attack"}, inplace = True)
	
	combined_pd = pd.concat([pred_pd, targ_pd], axis=1, join_axes=[pred_pd.index])
	final_pd = combined_pd.drop(combined_pd[(combined_pd.targ_nonattack == 0.0) & (combined_pd.targ_attack == 0.0)].index)
	pred = final_pd.iloc[:, :2].copy()
	targ = final_pd.iloc[:, 2:].copy()

	pred_attack = pred.iloc[:, 1].copy()
	targ_attack = targ.iloc[:, 1].copy()
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
	fig.savefig(FLAGS.model_name + '.png')
	#plt.show()


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
			
def output_layer(outputs, num_hidden, num_classes):

	final_output = tf.layers.dense(
			inputs= outputs,
			units= num_classes,
			activation= tf.nn.softmax,
			name="output_layer")

	
	return final_output

def apply_attention(Input_X, pos_enc, keep_prob):
	
	enc_outputs, enc_attention_weights = att.encode(pos_enc, FLAGS.num_hidden, 1)	
	dec_outputs, dec_attention_weights = att.decode(enc_outputs, Input_X, keep_prob, FLAGS.num_hidden, FLAGS.MAXLEN)
	
	
	return dec_outputs, enc_attention_weights, dec_attention_weights 

def evaluate_valid(validX, validY,Input_X, Input_Y, keep_prob, sess, loss, final_output, enc_attention_weights = None, dec_attention_weights = None, atten_test = False):

	get_valid_batch = Create_batch(validX, validY, FLAGS.batch_size)
	num_valid_batch = get_valid_batch.get_num_batch()

	valid_batch_loss = 0
	for batch in range(num_valid_batch):
		mini_valid_x, mini_valid_y = get_valid_batch.nextbatch()
		if atten_test:
			valid_loss, valid_pred, enc_w, dec_w = sess.run([loss, final_output, enc_attention_weights, dec_attention_weights], 
															  feed_dict={Input_X: mini_valid_x,
																		 Input_Y: mini_valid_y,
																		 keep_prob: 1.0})
			if batch == 0:
				enc_w_init = enc_w
				dec_w_init = dec_w
			elif batch > 0 and batch < num_valid_batch - 1:
				enc_w_init = np.concatenate([enc_w_init, enc_w], axis =0)
				dec_w_init = np.concatenate([dec_w_init, dec_w], axis =0)
			elif batch == num_valid_batch -1:
				enc_w_init = np.concatenate([enc_w_init, enc_w], axis =0)
				dec_w_init = np.concatenate([dec_w_init, dec_w], axis =0)
				enc_attention_weights = enc_w_init
				dec_attention_weights = dec_w_init			
		else:
			valid_loss, valid_pred = sess.run([loss, final_output], 
											  feed_dict={Input_X: mini_valid_x,
														 Input_Y: mini_valid_y,
														 keep_prob: 1.0})
		valid_batch_loss += valid_loss
		if batch == 0:
			valid_pred_init = valid_pred
		else:
			valid_pred_init = np.concatenate([valid_pred_init, valid_pred], axis =0)
			
	valid_batch_loss = valid_batch_loss/num_valid_batch		

	
	
	return valid_batch_loss, valid_pred_init, enc_attention_weights, dec_attention_weights

def main(unused_args):

	num_classes = 2
	
	tf.reset_default_graph()
	Input_X = tf.placeholder( tf.float32, [None, FLAGS.MAXLEN, FLAGS.num_features])
	Input_Y = tf.placeholder( tf.float32, [None, FLAGS.MAXLEN, num_classes])
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	length_X = None 
	
	
	if FLAGS.atten:
		model_output = att.pos_encoding(Input_X, keep_prob, FLAGS.num_hidden, FLAGS.MAXLEN)
		model_output, enc_attention_weights, dec_attention_weights = apply_attention(Input_X, model_output, keep_prob)

	else:
		model_output = bi.bi_lstm(Input_X, length_X, FLAGS.num_hidden, FLAGS.MAXLEN, keep_prob)	

	final_output = output_layer(model_output, FLAGS.num_hidden, num_classes)
	
	loss= weighted_crossentropy(Input_Y, final_output)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	train_op = optimizer.minimize(loss)
	
	# Start training
	saver = tf.train.Saver()
	
	train_loss_list = []
	val_loss_list = []
	train_f1_list = []
	val_f1_list = []
	

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)	
	config = tf.ConfigProto(

			device_count = {'GPU': 1},
			gpu_options=gpu_options
		)
	#config = tf.ConfigProto(

	#		device_count = {'GPU': 1},
	#	)
	
	if FLAGS.train:	
		train_path = "/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/reduced_attacks/one_percent/normed_oversampled_final_w10o9_train.csv"
		train_data = input_data_old.Fixed_seq(train_path, None, None, None)
		#train_data = input_data_old.Fixed_seq(FLAGS.input_path, None, None, None)
		#trainX, trainY = train_data.get_input('Train', SlidingAttackLabel = False)
		trainX, trainY = train_data.get_pickle('train')
		
		#valid_path = "/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/wedattack/preprocess_one/normed_wedlorisselect_combined_w10o0_validation.csv"
		#valid_data = input_data_old.Fixed_seq(valid_path, None, None, None)
		valid_data = input_data_old.Fixed_seq(FLAGS.input_path, None, None, None)
		#validX, validY = valid_data.get_input('Validation', SlidingAttackLabel = False)
		validX, validY = valid_data.get_pickle('validation')

		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer()) # Run the initializer
			
			max_valid_f1 = 0
			np.random.seed(115) #set seed for batch shuffle
			for epoch in range(1, FLAGS.num_epoch+1):
				past = datetime.now()
				
				shuffled_x, shuffled_y  = shuffle_fixed_data(trainX, trainY)
				get_batch = Create_batch(shuffled_x, shuffled_y, FLAGS.batch_size)
				num_batch = get_batch.get_num_batch()
		
				batch_loss = 0	
				batch_f1 = 0	
				for batch in range(num_batch):
					minibatch_x, minibatch_y = get_batch.nextbatch()
					sess.run(train_op, feed_dict={Input_X: minibatch_x, 
												  Input_Y: minibatch_y, 
												  keep_prob: FLAGS.drop})

					train_loss, train_pred = sess.run([loss, final_output], 
												  feed_dict={Input_X: minibatch_x,
															 Input_Y: minibatch_y,
															 keep_prob: 1.0})
					threshold = 0.5	
					_train_precision,_train_recall,_train_f1,_support,_,_,_ = get_result(train_pred, minibatch_y, threshold)
					batch_loss += train_loss
					batch_f1 += _train_f1[1]

				train_loss = batch_loss/num_batch
				train_f1 = batch_f1/num_batch
												  

				valid_loss, valid_pred,_,_ = evaluate_valid(validX, validY,Input_X, Input_Y, keep_prob,sess, loss, final_output)	
				_valid_precision,_valid_recall,_valid_f1,_support,_,_,_ = get_result(valid_pred, validY, threshold)
				
				if _valid_f1[1] > max_valid_f1:
					saver.save(sess, FLAGS.save_path + FLAGS.model_name)
					max_valid_f1 = _valid_f1[1]
					
				train_loss_list.append(train_loss)
				val_loss_list .append(valid_loss)
				train_f1_list .append(train_f1)
				val_f1_list .append(_valid_f1[1])
				
				now = datetime.now()
				
				print"\nEpoch {}/{} - {:.1f}s".format(epoch, FLAGS.num_epoch, (now - past).total_seconds()) 
				
				print "train_loss: {} ".format(train_loss)
				print "val_loss: {} ".format(valid_loss)
				print "train_f1: %4f" %(train_f1)
				print "val_f1: %4f" %(_valid_f1[1])
				print "\n"

				total_epochs = FLAGS.num_epoch
				if epoch%100 == 0:
					plot_performance(train_loss_list, val_loss_list, train_f1_list, val_f1_list, FLAGS.model_name)

			print("Optimization Finished!")
			#plot_performance(train_loss_list, val_loss_list, train_f1_list, val_f1_list, FLAGS.model_name)
		



	if FLAGS.test:
		AttackatLeast = FLAGS.MAXLEN/1
		#AttackatLeast =70 
		#test_path = "/home/cuc/CrossFire-Detect/CrossFire-Detect/data/cicids/new_features/all_attacks/normed_allselect_combined_w10o9_test.csv"
		#test_data = input_data_old.Fixed_seq(test_path, None, None, None)
		test_data = input_data_old.Fixed_seq(FLAGS.input_path, None, None, None)
		#testX, testY = test_data.get_input('Test', SlidingAttackLabel = False)
		#_, testY2 = test_data.get_input('Test', SlidingAttackLabel = AttackatLeast)
		testX, testY = test_data.get_pickle('test')
		
		config = tf.ConfigProto(
			device_count = {'GPU': 0}
			)
		
		with tf.Session(config= config) as sess:
			saver.restore(sess,FLAGS.save_path + FLAGS.model_name)
		
			if FLAGS.atten:
				#test_pred, enc_w, dec_w  = sess.run([final_output, enc_attention_weights, dec_attention_weights], 
				#				  feed_dict={Input_X: testX,
				#							keep_prob: 1.0})
				#test_pred = sess.run(final_output, 
				#				  feed_dict={Input_X: testX,
				#							keep_prob: 1.0})
				_, test_pred, enc_w, dec_w = evaluate_valid(testX, testY,Input_X, Input_Y, keep_prob,sess, loss, final_output, enc_attention_weights, dec_attention_weights, atten_test = True)	
			else:
				#test_pred = sess.run(final_output, 
				#				  feed_dict={Input_X: testX,
				#							keep_prob: 1.0})
				
				_, test_pred,_,_ = evaluate_valid(testX, testY,Input_X, Input_Y, keep_prob,sess, loss, final_output)	

			threshold = 0.5
			_test_precision,_test_recall,_test_f1,_support, pred, targ, conf = get_result(test_pred, testY, threshold)
			print '\n'
			print " - test_f1: (%f,%f) - test_precision: (%f,%f) - test_recall (%f,%f) - test_support(%f,%f)" % \
			(_test_f1[0],_test_f1[1], _test_precision[0],_test_precision[1], _test_recall[0],_test_recall[1], _support[0],_support[1])
			print conf

			#pdb.set_trace()
#			test_pred2 = convert_cicids.convert_result(test_pred, threshold, AttackatLeast)
#			_test_precision,_test_recall,_test_f1,_support, pred, targ, conf = convert_cicids.get_result(test_pred2, testY2)
#
#			print 'Test For Windows'
#			print " - test_f1: (%f,%f) - test_precision: (%f,%f) - test_recall (%f,%f) - test_support(%f,%f)" % \
#			(_test_f1[0],_test_f1[1], _test_precision[0],_test_precision[1], _test_recall[0],_test_recall[1], _support[0],_support[1])
#			print conf
#			
			print 'atten:{}, keep_prob:{}, bilstm_hidden:{}, features:{}, MAXLEN:{}, epoch:{}, batch:{}, model:{}, threshold:{}, AttackatLeast:{}'\
				  .format(FLAGS.atten, FLAGS.drop, FLAGS.num_hidden, FLAGS.num_features, FLAGS.MAXLEN,\
							FLAGS.num_epoch, FLAGS.batch_size, FLAGS.model_name, threshold, AttackatLeast)

			if FLAGS.plot:

				sample_idx = FLAGS.sample_idx
				
				_test_precision,_test_recall,_test_f1,_support, pred, targ, conf = get_result(
				test_pred[sample_idx:sample_idx + 1, :, :],
				 testY[sample_idx:sample_idx + 1, :, :], threshold)
				print '\n'
				print "Sample - test_f1: (%f,%f) - test_precision: (%f,%f) - test_recall (%f,%f) - test_support(%f,%f)" % \
				(_test_f1[0],_test_f1[1], _test_precision[0],_test_precision[1], _test_recall[0],_test_recall[1], _support[0],_support[1])
				print conf

				prediction = test_pred[sample_idx][:, [0]].T
				truth = testY[sample_idx][:, [0]].T
				enc = enc_w[sample_idx, :, :]
				dec = dec_w[sample_idx, :, :]
				
				fig = plt.figure(figsize=(40, 20))
				gs = gridspec.GridSpec(6, 4, hspace=0.4, wspace=0.3)
				
				ax = plt.Subplot(fig, gs[0, : ])
				sns.heatmap(data= truth, cbar=False, linewidth=0.2, vmin= 0, vmax= 1,  ax=ax)
				ax.set_title("Truth_WebAttackBruteForce", fontsize = 10)
				fig.add_subplot(ax)
				
				ax = plt.Subplot(fig, gs[1, : ])
				sns.heatmap(data=prediction, cbar=False, linewidth=0.2, vmin= 0, vmax= 1,  ax=ax)
				ax.set_title("Prediction_WebAttackBruteFoce", fontsize = 10)
				fig.add_subplot(ax)

				ax = plt.Subplot(fig, gs[2:, :2])
				sns.heatmap(data=enc, cbar=True, linewidth=0.2, ax=ax)
				ax.set_title("Attention_weights", fontsize = 15)
				fig.add_subplot(ax)

				ax = plt.Subplot(fig, gs[2:, 2:])
				sns.heatmap(data=dec, cbar=True, linewidth=0.2, ax=ax)
				ax.set_title("decoder_weights", fontsize = 10)
				fig.add_subplot(ax)
				fig.savefig('attention_weights.png')
				print "fig saved!"
				plt.show()


if __name__ == "__main__":
	app.run(main)
