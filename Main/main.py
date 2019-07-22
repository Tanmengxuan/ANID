import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

from utils import *
from os import mkdir,environ
import time
from datetime import datetime

from absl import flags
from absl import app

import pdb

import attention as att
import bilstm as bi

import h5py

FLAGS = flags.FLAGS
# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("test_multi", False, "Test multiple trained instances of a model on different attacks")

# Models parameters
flags.DEFINE_bool("atten", False, "Train ANID (attention) model")
flags.DEFINE_bool("bilstm", False, "Train bilstm model")
flags.DEFINE_float("drop", 0.9, "keep_prob for dropout")
flags.DEFINE_integer("anid_hidden", 100, "hidden size of anid")
flags.DEFINE_integer("bilstm_hidden", 600, "hidden size of bilstm")
flags.DEFINE_integer("num_features", 19, "feature size of input")
flags.DEFINE_integer("seq_len", 10, "window size of input data")

# Training parameters
flags.DEFINE_integer("num_epoch", 3000, "number of training epochs")
flags.DEFINE_integer("batch_size", 256, "batch size of training datasets")

# Paths
flags.DEFINE_string("model_name", "cicids", "name of saved model")
flags.DEFINE_string("save_path", "../checkpoints/", "path of saved model")
flags.DEFINE_string("input_path", "../data/normed_w10o9_*",
"input path of data") 

def evaluate_test(testX, testY,Input_X, Input_Y, keep_prob, sess, loss, final_output):

	get_test_batch = Create_batch(testX, testY, FLAGS.batch_size)
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

def main(unused_args):
	#pdb.set_trace()

	num_classes = 2
	
	tf.reset_default_graph()
	Input_X = tf.placeholder( tf.float32, [None, FLAGS.seq_len, FLAGS.num_features])
	Input_Y = tf.placeholder( tf.float32, [None, FLAGS.seq_len, num_classes])
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	if FLAGS.atten:
		#pdb.set_trace()
		model_output_pos = att.pos_encoding(Input_X, keep_prob, FLAGS.anid_hidden, FLAGS.seq_len)
		model_output, enc_attention_weights, dec_attention_weights = att.apply_attention(Input_X, model_output_pos, FLAGS.anid_hidden, FLAGS.seq_len, keep_prob)

	elif FLAGS.bilstm:
		model_output = bi.bi_lstm(Input_X, FLAGS.bilstm_hidden, FLAGS.seq_len, keep_prob)	

	num_hidden = FLAGS.anid_hidden if FLAGS.atten else FLAGS.bilstm_hidden
	final_output = output_layer(model_output, num_hidden, num_classes)
	
	loss= weighted_crossentropy(Input_Y, final_output)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	train_op = optimizer.minimize(loss)
	
	# Start training
	saver = tf.train.Saver()
	
	train_loss_list = []
	val_loss_list = []
	train_f1_list = []
	val_f1_list = []
	

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)	
	config = tf.ConfigProto(

			device_count = {'GPU': 1},
			gpu_options=gpu_options
		)
	
	if FLAGS.train:	
		#pdb.set_trace()

		data = Inputter(FLAGS.input_path)
		trainX, trainY = data.load_data('train')
		validX, validY = data.load_data('valid')

		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer()) # Run the initializer
			
			#print tf.trainable_variables()
			#print [n.name for n in tf.get_default_graph().as_graph_def().node]

		#with tf.Session(config= config) as sess: #continue training with a saved model
		#	saver.restore(sess,FLAGS.save_path + 'cicids63')
			
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
												  keep_prob: FLAGS.drop,
												})

					train_loss, train_pred = sess.run([loss, final_output], 
												  feed_dict={Input_X: minibatch_x,
															 Input_Y: minibatch_y,
															 keep_prob: 1.0,
															})
					threshold = 0.5	
					_train_precision,_train_recall,_train_f1,_support,_,_,_ = get_result(train_pred, minibatch_y, threshold)
					batch_loss += train_loss
					batch_f1 += _train_f1[1]

				train_loss = batch_loss/num_batch
				train_f1 = batch_f1/num_batch
												  
				valid_loss, valid_pred = evaluate_test(validX, validY,Input_X, Input_Y, keep_prob,sess, loss, final_output)	
				_valid_precision,_valid_recall,_valid_f1,_support,_,_,_ = get_result(valid_pred, validY, threshold)
				
				if _valid_f1[1] > max_valid_f1:
					saver.save(sess, FLAGS.save_path + FLAGS.model_name)
					max_valid_f1 = _valid_f1[1]
					
				train_loss_list.append(train_loss)
				val_loss_list .append(valid_loss)
				train_f1_list .append(train_f1)
				val_f1_list .append(_valid_f1[1])
				
				now = datetime.now()
				print ("\nEpoch {}/{} - {:.1f}s".format(epoch, FLAGS.num_epoch, (now - past).total_seconds())) 
				
				print ("train_loss: {:.6f} ".format(train_loss))
				print ("val_loss: {:.6f} ".format(valid_loss))
				print ("train_f1: {:.6f}" .format(train_f1))
				print ("val_f1: {:.6f}" .format(_valid_f1[1]))
				print ("\n")

				total_epochs = FLAGS.num_epoch
				if epoch%100 == 0:
					plot_performance(train_loss_list, val_loss_list, train_f1_list, val_f1_list, FLAGS.model_name)

			print("Optimization Finished!")
			plot_performance(train_loss_list, val_loss_list, train_f1_list, val_f1_list, FLAGS.model_name)
			h5 = h5py.File('../train_records/' + FLAGS.model_name + '_records.h5', 'w')
			h5.create_dataset('train_loss', data = train_loss_list)
			h5.create_dataset('val_loss', data = val_loss_list)
			h5.create_dataset('train_f1', data = train_f1_list)
			h5.create_dataset('val_f1', data = val_f1_list)



	if FLAGS.test:

		data = Inputter(FLAGS.input_path)
		testX, testY = data.load_data('test')

		#config = tf.ConfigProto(
		#	device_count = {'GPU': 1}
		#	)
		
		with tf.Session(config= config) as sess:
			saver.restore(sess,FLAGS.save_path + FLAGS.model_name)
			#print (tf.trainable_variables())
			get_total_para(tf.trainable_variables())
			start_time_init = time.time()

			_, test_pred =  evaluate_test(testX, testY,Input_X, Input_Y, keep_prob,sess, loss, final_output)	

			print ('time for inference: ', time.time() - start_time_init)
			threshold = 0.5
			_test_precision,_test_recall,_test_f1,_support, pred, targ, conf = get_result(test_pred, testY, threshold)
			print ('\n')
			print (" - test_f1: %f - test_precision: %f - test_recall %f " % \
			(_test_f1[1], _test_precision[1], _test_recall[1]))
			print (conf)

			print ('atten:{}, keep_prob:{}, hidden_size:{}, features:{}, seq_len:{}, epoch:{}, batch:{}, model:{}, threshold:{}'\
				.format(FLAGS.atten, FLAGS.drop, num_hidden, FLAGS.num_features, FLAGS.seq_len,\
				FLAGS.num_epoch, FLAGS.batch_size, FLAGS.model_name, threshold))
			

			if FLAGS.test_multi:

				multi = Multi()
				models_list = ['cicids74', 'cicids94', 'cicids95', 'cicids96', 'cicids97', 'cicids98', 'cicids99', 'cicids100', 'cicids101', 'cicids102']	
				#models_list = [FLAGS.model_name]

				attack_list = ['dos', 'ddos', 'infl', 'brute', 'port', 'web', 'botnet']
				obj_list = []
				for attack in attack_list:
					obj_list.append(Attacks(attack))

				for model_name in models_list:
					with tf.Session(config= config) as sess:

						saver.restore(sess,FLAGS.save_path + model_name)
						start_time = time.time()

						_, test_pred = evaluate_test(testX, testY,Input_X, Input_Y, keep_prob,sess, loss, final_output)
						
						print ('time for inference: ', time.time() - start_time)
						threshold = 0.5
						_test_precision,_test_recall,_test_f1,_support, pred, targ, conf = get_result(test_pred, testY, threshold)
						print ('\n')
						print (" - test_f1: %f - test_precision: %f - test_recall %f " % \
						(_test_f1[1], _test_precision[1], _test_recall[1]))
						print (conf)
						
						multi.compute(conf)

						fp_all = conf[0][1]
						for obj in obj_list:
							obj.compute_all_attacks(fp_all, test_pred, testY)

				for obj in obj_list:
					obj.get_result()
				
				multi.get_result()


if __name__ == "__main__":
	app.run(main)
