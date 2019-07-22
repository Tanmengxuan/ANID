import tensorflow as tf
import numpy as np
import h5py
import pdb

def attention(query, key, value, keep_prob):
	# Equation 1 in Vaswani et al. (2017)
	# 	Scaled dot product between Query and Keys
	output = tf.matmul(query, key, transpose_b=True) / (tf.cast(tf.shape(query)[2], tf.float32) ** 0.5)
	# 	Softmax to get attention weights
	attention_weights = tf.nn.softmax(output)
	# 	Multiply weights by Values
	weighted_sum = tf.matmul(attention_weights, value)
	# Following Figure 1 and Section 3.1 in Vaswani et al. (2017)
	# 	Residual connection ie. add weighted sum to original query
	output = weighted_sum + query
	# 	Layer normalization
	output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)

	return output, attention_weights


def pos_encoding(input_x, keep_prob, num_hidden, MAXLEN):

	#pass initial input through dense to become embedded
	embedded_input = tf.layers.dense(
					inputs = input_x,
					units = num_hidden,
					activation = None,
					name = 'input_dense')
	
	positional_encoding = tf.Variable(
				initial_value = tf.zeros((1, MAXLEN, num_hidden)),
				trainable = True,
				dtype = tf.float32,
				name = 'pos_encoding')
	
	positional_input = embedded_input + positional_encoding
	
	positional_input = tf.nn.dropout(
				positional_input,
				keep_prob,
				name = "sum_positional_dropout")
	
	return positional_input


def sdsa(encoding, keep_prob):
	
	encoding, enc_attention_weights = attention(encoding, encoding, encoding, keep_prob)

	print ("sdsa_out", encoding)

	return encoding, enc_attention_weights

def ffn(encoding, num_hidden):

	dense = tf.layers.dense(
			inputs=encoding,
			units= num_hidden*8,
			activation=tf.nn.relu,
			name="ffn_out",
			#name="encoder_layer1_dense1",
	)
	
	print ("ffn_out", dense)
	
	encoding += tf.layers.dense(
			inputs=dense,
			units= num_hidden,
			activation=None,
			name="ffn_add",
			#name="encoder_layer1_dense2",
	)
	
	print ("ffn_add", encoding)
	
	encoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)

	return encoding

def ffn_2(encoding, num_hidden):

	dense = tf.layers.dense(
			inputs=encoding,
			units= num_hidden*8,
			activation=tf.nn.relu,
			name="ffn_out_2",
			#name="encoder_layer1_dense1",
	)
	
	print ("ffn_out_2", dense)
	
	encoding += tf.layers.dense(
			inputs=dense,
			units= num_hidden,
			activation=None,
			name="ffn_add_2",
			#name="encoder_layer1_dense2",
	)
	
	print ("ffn_add_2", encoding)
	
	encoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)

	return encoding

def sda(encoding, num_hidden, initial_input, MAXLEN, keep_prob):
	
	decoder_input = tf.Variable(
					initial_value = np.zeros((1, MAXLEN, num_hidden)),
					trainable=True,
					dtype=tf.float32,
					name="sda_query",
					#name="decoder_input",
	 )
				
	print ("sda_query", decoder_input)
	
	decoded, decoder_attention_weights = attention(
	tf.tile(decoder_input, multiples=tf.concat(([tf.shape(initial_input)[0]], [1], [1]), axis=0)),
	encoding,
	encoding,
	keep_prob,
	)
	
				
	print ("sda_out", decoded)
	print ("sda_out_weights", decoder_attention_weights)
	
	return decoded, decoder_attention_weights


def sda_2(encoding, num_hidden, initial_input, MAXLEN, keep_prob):
	
	decoder_input = tf.Variable(
					initial_value = np.zeros((1, MAXLEN, num_hidden)),
					trainable=True,
					dtype=tf.float32,
					name="sda_query_2",
					#name="decoder_input",
	 )
				
	print ("sda_query_2", decoder_input)
	
	decoded, decoder_attention_weights = attention(
	tf.tile(decoder_input, multiples=tf.concat(([tf.shape(initial_input)[0]], [1], [1]), axis=0)),
	encoding,
	encoding,
	keep_prob,
	)
	
	
	print ("sda_out_2", decoded)
	print ("sda_out_weights_2", decoder_attention_weights)
	
	return decoded, decoder_attention_weights


def apply_attention(initial_input, model_output, num_hidden, maxlen, keep_prob):

	''' function used for combination study'''

	#sdsa_out, sdsa_attention_weights = sdsa(model_output, keep_prob)
	#ffn_out = ffn(sdsa_out, num_hidden)
	#sda_out, sda_attention_weights = sda(ffn_out, num_hidden, initial_input, maxlen, keep_prob)

	#sda_out, sda_attention_weights = sda(model_output, num_hidden, initial_input, maxlen, keep_prob)
	#ffn_out = ffn(sda_out, num_hidden)
	#sdsa_out, sdsa_attention_weights = sdsa(ffn_out, keep_prob)

	#ffn_out = ffn(model_output, num_hidden)

	#sda_out, sda_attention_weights = sda(model_output, num_hidden, initial_input, maxlen, keep_prob)

	#ffn_out = ffn(model_output, num_hidden)
	#sdsa_out, sdsa_attention_weights = sdsa(ffn_out, keep_prob)

	#ffn_out = ffn(model_output, num_hidden)
	#sda_out, sda_attention_weights = sda(ffn_out, num_hidden, initial_input, maxlen, keep_prob)

	#sda_out, sda_attention_weights = sda(model_output, num_hidden, initial_input, maxlen, keep_prob)
	#ffn_out = ffn(sda_out, num_hidden)

	#sdsa_out, sdsa_attention_weights = sdsa(model_output, keep_prob)
	#sda_out, sda_attention_weights = sda(sdsa_out, num_hidden, initial_input, maxlen, keep_prob)

	#sda_out, sda_attention_weights = sda(model_output, num_hidden, initial_input, maxlen, keep_prob)
	#sdsa_out, sdsa_attention_weights = sdsa(sda_out, keep_prob)

	#sdsa_out, sdsa_attention_weights = sdsa(model_output, keep_prob)
	#sda_out, sda_attention_weights = sda(sdsa_out, num_hidden, initial_input, maxlen, keep_prob)
	#ffn_out = ffn(sda_out, num_hidden)

	#sda_out, sda_attention_weights = sda(model_output, num_hidden, initial_input, maxlen, keep_prob)
	#sdsa_out, sdsa_attention_weights = sdsa(sda_out, keep_prob)
	#ffn_out = ffn(sdsa_out, num_hidden)

	ffn_out = ffn(model_output, num_hidden)
	sda_out, sda_attention_weights = sda(ffn_out, num_hidden, initial_input, maxlen, keep_prob)
	sdsa_out, sdsa_attention_weights = sdsa(sda_out, keep_prob)

	#ffn_out = ffn(model_output, num_hidden)
	#sdsa_out, sdsa_attention_weights = sdsa(ffn_out, keep_prob)
	#sda_out, sda_attention_weights = sda(sdsa_out, num_hidden, initial_input, maxlen, keep_prob)

	#ffn_out = ffn(model_output, num_hidden)
	#sda_out_1, sda_attention_weights_1 = sda(ffn_out, num_hidden, initial_input, maxlen, keep_prob)
	#sda_out, sda_attention_weights_2 = sda_2(sda_out_1, num_hidden, initial_input, maxlen, keep_prob)

	#ffn_out_1 = ffn(model_output, num_hidden)
	#sda_out, sda_attention_weights = sda(ffn_out_1, num_hidden, initial_input, maxlen, keep_prob)
	#ffn_out = ffn_2(sda_out, num_hidden)

	#return sda_out, sdsa_attention_weights, sda_attention_weights
	return sdsa_out, sdsa_attention_weights, sda_attention_weights
	#return ffn_out, sdsa_attention_weights, sda_attention_weights
	#return ffn_out
	#return ffn_out, sda_attention_weights, sda_attention_weights
	#return sda_out, sda_attention_weights, sda_attention_weights
	#return sdsa_out, sdsa_attention_weights, sdsa_attention_weights
	#return sda_out, sda_attention_weights_1, sda_attention_weights_2
