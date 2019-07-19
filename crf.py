import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


import keras
from keras.models import Model, Input
from keras.layers import LSTM,  Dense, Dropout, Bidirectional, Masking
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model
from utils import *
import h5py
import time
from datetime import datetime


#trainX
#trainY
#validX
#validY

def train_crf(trainX, trainY, validX, validY, save_path):


	feature_size = 19
	dense_size = 50
	num_classes = 2
	drop = 0.1
	batch_size = 256 
	num_epochs = 2000


	#model_input = Input(batch_shape = (None, None, feature_size))
	model_input = Input(shape = (10, feature_size))
	model_input_mask = Masking(mask_value = 999)(model_input) # dummy mask to ensure loss is positive
	#dense0 = Dense(dense_size, activation = None)(model_input_mask)
	#dense0 = Dropout(drop)(dense0)

	#dense0 = Bidirectional(LSTM(units=300, return_sequences=True,
	#						   dropout=drop))(model_input_mask) 
	crf = CRF(num_classes)
	#model_output = crf(dense0)
	model_output = crf(model_input_mask)

	model = Model(model_input, model_output)

	model.compile(optimizer="adam", loss=crf_loss, metrics= [crf_viterbi_accuracy])
	print(model.summary())


	train_loss_list = []
	valid_loss_list = []
	#train_f1_list = []
	valid_f1_list = []
	max_valid_f1 = 0

	for epoch in range(1, num_epochs + 1):

		past = datetime.now()

		shuffled_x, shuffled_y  = shuffle_fixed_data(trainX, trainY)
		history = model.fit(shuffled_x, shuffled_y, validation_data = (validX, validY), batch_size=batch_size, epochs=1, verbose=1)

		#train_pred = model.predict(shuffled_x)
		valid_pred = model.predict(validX)

		threshold = 0.5	
		#_train_precision,_train_recall,_train_f1,_support,_,_,_ = get_result(train_pred, shuffled_y, threshold)
		_valid_precision,_valid_recall,_valid_f1,_support,_,_,_ = get_result(valid_pred, validY, threshold)

		train_loss = history.history['loss'][0]
		valid_loss = history.history['val_loss'][0]

		if _valid_f1[1] > max_valid_f1:
			model.save(save_path)
			max_valid_f1 = _valid_f1[1]


		train_loss_list.append(train_loss)
		valid_loss_list .append(valid_loss)
		#train_f1_list .append(_train_f1[1])
		valid_f1_list .append(_valid_f1[1])

		now = datetime.now()
		print ("\nEpoch {}/{} - {:.1f}s".format(epoch, num_epochs, (now - past).total_seconds())) 
		
		print ("train_loss: {:.6f} ".format(train_loss))
		print ("val_loss: {:.6f} ".format(valid_loss))
		#print ("train_f1: {:.6f}" .format(_train_f1[1]))
		print ("valid_f1: {:.6f}" .format(_valid_f1[1]))
		print ("\n")

		if epoch%100 == 0:
			plot_performance_crf(train_loss_list, valid_loss_list, valid_f1_list, model_name)

	print("Optimization Finished!")
	plot_performance_crf(train_loss_list, valid_loss_list, valid_f1_list, model_name)
	h5 = h5py.File('h5files/' + model_name + '_records.h5', 'w')
	h5.create_dataset('train_loss', data = train_loss_list)
	h5.create_dataset('valid_loss', data = valid_loss_list)
	#h5.create_dataset('train_f1', data = train_f1_list)
	h5.create_dataset('valid_f1', data = valid_f1_list)


input_path = '/home/tmx/cicids_alfonso/normed_allreducedselect_final_w10o9_*'
data = Inputter(input_path)
trainX, trainY = data.load_data('train')
validX, validY = data.load_data('valid')
testX, testY = data.load_data('test')



model_name = 'crf3.h5'
save_path = 'lstm_runs/cicids/wedattack/new_features/' + model_name

#train_crf(trainX, trainY, validX, validY, save_path)


custom_objects={'CRF': CRF,
                'crf_loss': crf_loss,
                'crf_viterbi_accuracy': crf_viterbi_accuracy}

test_model = load_model(save_path, custom_objects= custom_objects)

test_pred = test_model.predict(testX)
threshold = 0.5
_test_precision,_test_recall,_test_f1,_support, pred, targ, conf = get_result(test_pred, testY, threshold)
print ('\n')
print (" - test_f1: (%f,%f) - test_precision: (%f,%f) - test_recall (%f,%f) - test_support(%f,%f)" % \
(_test_f1[0],_test_f1[1], _test_precision[0],_test_precision[1], _test_recall[0],_test_recall[1], _support[0],_support[1]))
print (conf)

