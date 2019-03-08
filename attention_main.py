import tensorflow as tf
import numpy as np

def attention(query, key, value):
	# Equation 1 in Vaswani et al. (2017)
	# 	Scaled dot product between Query and Keys
	output = tf.matmul(query, key, transpose_b=True) / (tf.cast(tf.shape(query)[2], tf.float32) ** 0.5)
	# 	Softmax to get attention weights
	#output = apply_mask(output, Mask)
	attention_weights = tf.nn.softmax(output)
	#attention_weights = apply_mask(attention_weights, Mask)
	# 	Multiply weights by Values
	weighted_sum = tf.matmul(attention_weights, value)
	# Following Figure 1 and Section 3.1 in Vaswani et al. (2017)
	# 	Residual connection ie. add weighted sum to original query
	output = weighted_sum + query
	# 	Layer normalization
	output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
	#return output, attention_weights, attention_weights_unmasked
	return output, attention_weights

def encode(encoding, num_hidden, enc_layers):
	
	for i in range(enc_layers):
		encoding, enc_attention_weights = attention(encoding, encoding, encoding)

		print "encoding", encoding
		
		dense = tf.layers.dense(
				inputs=encoding,
				units= num_hidden*2*2*2*2,
				activation=tf.nn.relu,
				name="encoder_layer{}_dense1".format(i + 1)
		)
		
		#dense = tf.nn.dropout(
		#		dense,
		#		1) 
		print "encoder_layer{}_dense1".format(i + 1), dense
		
		#dense += tf.layers.dense(
		#		inputs=dense,
		#		units= num_hidden*2*2*2,
		#		activation=None,
		#		name="encoder_laye2__dense2"
		#)
		#
		#print "encoder_layer2_dense2", encoding
		
		encoding += tf.layers.dense(
				inputs=dense,
				units= num_hidden*2,
				activation=None,
				name="encoder_layer{}_dense2".format(i + 1)
		)
		
		print "encoder_layer{}_dense2".format(i + 1), encoding
		
		encoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)

	return encoding, enc_attention_weights

def decode(encoding, initial_input, keep_prob, num_hidden, MAXLEN):
	
	#print "encoding", encoding
	#
	#dense = tf.layers.dense(
	#		inputs=encoding,
	#		units= num_hidden*2*2*2,
	#		activation=tf.nn.tanh,
	#		name="encoder_layer1_dense1"
	#)
	#
	##dense = tf.nn.dropout(
	##		dense,
	##		1) 
	#print "encoder_layer1_dense1", dense
	#
	##dense += tf.layers.dense(
	##		inputs=dense,
	##		units= num_hidden*2*2*2,
	##		activation=None,
	##		name="encoder_laye2__dense2"
	##)
	##
	##print "encoder_layer2_dense2", encoding
	#
	#encoding += tf.layers.dense(
	#		inputs=dense,
	#		units= num_hidden*2,
	#		activation=None,
	#		name="encoder_laye3__dense3"
	#)
	#
	#print "encoder_layer3_dense3", encoding
	#
	#encoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)
	
	decoder_input = tf.Variable(
					initial_value = np.zeros((1, MAXLEN, num_hidden*2)),
					trainable=True,
					dtype=tf.float32,
					name="decoder_input",
	 )
				
	print "decoder_input", decoder_input

	decoded, decoder_attention_weights = attention(
	tf.tile(decoder_input, multiples=tf.concat(([tf.shape(initial_input)[0]], [1], [1]), axis=0)),
	encoding,
	encoding,
	)

	#decoded, decoder_attention_weights = multihead_attention(
	#tf.tile(decoder_input, multiples=tf.concat(([tf.shape(initial_input)[0]], [1], [1]), axis=0)),
	#encoding,
	#encoding,
	#num_hidden*2,
	#4
	#)
				
	print "decoded", decoded
	print "decoder_attention_weights", decoder_attention_weights
	
	return decoded, decoder_attention_weights


def pos_encoding(input_x, keep_prob, num_hidden, MAXLEN):

	#pass initial input through dense to become embedded
	embedded_input = tf.layers.dense(
					inputs = input_x,
					units = num_hidden*2,
					activation = None,
					name = 'input_dense')

	embedded_input = tf.nn.dropout(
						embedded_input,
						keep_prob,
						name = "embedded_input_dropout") 

	positional_encoding = tf.Variable(
						initial_value = tf.zeros((1, MAXLEN, num_hidden*2)),
						trainable = True,
						dtype = tf.float32,
						name = 'pos_encoding')

	positional_encoding = tf.nn.dropout(
						positional_encoding,
						keep_prob,
						name = "positional_encoding_dropout") 

	positional_input = embedded_input + positional_encoding

	return positional_input

def multihead_attention( query, key, value, hidden, h=4):
	W_query = tf.Variable(
		initial_value=tf.random_normal((hidden, hidden), stddev=1e-2),
		trainable=True,
		dtype=tf.float32,
	)
	W_key = tf.Variable(
		initial_value=tf.random_normal((hidden, hidden), stddev=1e-2),
		trainable=True,
		dtype=tf.float32,
	)
	W_value = tf.Variable(
		initial_value=tf.random_normal((hidden, hidden), stddev=1e-2),
		trainable=True,
		dtype=tf.float32,
	)
	W_output = tf.Variable(
		initial_value=tf.random_normal((hidden, hidden), stddev=1e-2),
		trainable=True,
		dtype=tf.float32,
	)
	multi_query = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(query, [-1, hidden]), W_query), [-1, 1, tf.shape(query)[1], h, int(hidden/h)]), axis=3), axis= 1)
	multi_key = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(key, [-1, hidden]), W_key), [-1, 1, tf.shape(key)[1], h, int(hidden/h)]), axis=3), axis= 1)
	multi_value = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(value, [-1, hidden]), W_value), [-1, 1, tf.shape(value)[1], h, int(hidden/h)]), axis=3), axis= 1)
	dotp = tf.matmul(multi_query, multi_key, transpose_b=True) / (tf.cast(tf.shape(multi_query)[-1], tf.float32) ** 0.5)
	attention_weights = tf.nn.softmax(dotp)
	weighted_sum = tf.matmul(attention_weights, multi_value)
	weighted_sum = tf.concat(tf.unstack(weighted_sum, axis=1), axis=-1)
	
	multihead = tf.reshape(tf.matmul(tf.reshape(weighted_sum, [-1, hidden]), W_output), [-1, tf.shape(query)[1], hidden])
	output = multihead + query
	output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
	return output, attention_weights

