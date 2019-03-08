import tensorflow as tf
from tensorflow.contrib import rnn

def variable_length(sequence):
	
	used = tf.sign(tf.reduce_max(tf.abs(sequence), axis = 2))
	length = tf.reduce_sum(used, 1)
	length = tf.cast(length, tf.int32)
	
	return length


def bi_lstm(Input_X, length_X, num_hidden, MAXLEN, keep_prob):
	

	x = tf.unstack(Input_X, MAXLEN, 1)
	
	
	lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	lstm_fw_cell= rnn.DropoutWrapper(lstm_fw_cell,input_keep_prob=keep_prob, output_keep_prob=keep_prob)
	
	lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	lstm_bw_cell= rnn.DropoutWrapper(lstm_bw_cell,input_keep_prob=keep_prob, output_keep_prob=keep_prob)
	
	outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
												  dtype=tf.float32,
												  sequence_length = length_X)
	#outputs, _ = tf.nn.static_rnn(lstm_fw_cell, x, dtype=tf.float32)	
	outputs = tf.stack(outputs,axis = 1) 

	return outputs

