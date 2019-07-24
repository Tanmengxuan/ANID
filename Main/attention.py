import tensorflow as tf
import numpy as np
import h5py
import pdb

def attention(query, key, value):

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


def pos_encoding(input_x, num_hidden, seq_len, keep_prob):

	#pass initial input through dense to become embedded
	embedded_input = tf.layers.dense(
				inputs = input_x,
				units = num_hidden,
				activation = None,
				name = 'input_dense')

	#giving inputs positional information
	positional_encoding = tf.Variable(
				initial_value = tf.zeros((1, seq_len, num_hidden)),
				trainable = True,
				dtype = tf.float32,
				name = 'pos_encoding')
	
	positional_input = embedded_input + positional_encoding
	
	positional_input = tf.nn.dropout(
				positional_input,
				keep_prob,
				name = "sum_positional_dropout")
	
	return positional_input


def sdsa(encoding):
	
	'''SDSA module'''

	encoding, enc_attention_weights = attention(encoding, encoding, encoding)

	print ("sdsa_out", encoding)

	return encoding, enc_attention_weights

def ffn(encoding, num_hidden):
	
	'''FF module'''

	dense = tf.layers.dense(
			inputs=encoding,
			units= num_hidden*8,
			activation=tf.nn.relu,
			name="ffn_out",
	)
	
	print ("ffn_out", dense)
	
	encoding += tf.layers.dense(
			inputs=dense,
			units= num_hidden,
			activation=None,
			name="ffn_add",
	)
	
	print ("ffn_add", encoding)
	
	encoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)

	return encoding

def sda(encoding, num_hidden, initial_input, seq_len):

	'''SDA module'''

	decoder_input = tf.Variable(
				initial_value = np.zeros((1, seq_len, num_hidden)),
				trainable=True,
				dtype=tf.float32,
				name="sda_query",
	 )
				
	print ("sda_query", decoder_input)
	
	decoded, decoder_attention_weights = attention(
	tf.tile(decoder_input, multiples=tf.concat(([tf.shape(initial_input)[0]], [1], [1]), axis=0)),
	encoding,
	encoding,
	)
	
				
	print ("sda_out", decoded)
	print ("sda_out_weights", decoder_attention_weights)
	
	return decoded, decoder_attention_weights


def apply_attention(model_output, initial_input, num_hidden, seq_len):

	''' a function which can be used for combination study'''

	ffn_out = ffn(model_output, num_hidden)
	sda_out, sda_attention_weights = sda(ffn_out, num_hidden, initial_input, seq_len)
	sdsa_out, sdsa_attention_weights = sdsa(sda_out)

	return sdsa_out, sdsa_attention_weights, sda_attention_weights
