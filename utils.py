import tensorflow as tf
import pdb
from keras import backend as K 
import numpy as np
import pandas as pd
import pdb
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

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

def weighted_crossentropy_nn(y_true, y_pred):
	#pdb.set_trace()
	output = y_pred
	output /= tf.reduce_sum(y_pred, -1, True)
	output = tf.clip_by_value(output, 1e-5, 1. - 1e-5)
	#xent = - tf.reduce_sum(y_true * tf.log(output), -1)
	# pos_weight = 5.87
	# xent *= y_true[:,:,1] * pos_weight + y_true[:,:,0] * 1
	#xent *=   y_true[:,:,1] * (tf.cast(tf.reduce_sum(y_true[:, :, 0]), tf.float32) / tf.reduce_sum(y_true[:, :, 1])) + y_true[:, :, 0] * 1 
	pos_weight = tf.cast(tf.reduce_sum(y_true[:, 0]), tf.float32) / (tf.reduce_sum(y_true[:, 1]) + 1e-5)
	loss = y_true[:,1]*pos_weight*tf.log(output)[:,1] + y_true[:,0]*1*tf.log(output)[:,0]
	xent = - tf.reduce_mean(loss)
	return xent

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

def get_result(test_pred, testY, threshold):
    
	#pred = test_pred.reshape(-1, test_pred.shape[2])
	pred = test_pred > threshold
	pred = pred.astype(int)
	pred_pd = pd.DataFrame(pred)
	pred_pd.rename(columns = {0:"pred_nonattack", 1:"pred_attack"}, inplace = True)
	
	#targ = testY.reshape( -1, testY.shape[2])
	targ_pd = pd.DataFrame(testY)
	targ_pd.rename(columns = {0:"targ_nonattack", 1:"targ_attack"}, inplace = True)
	
	combined_pd = pd.concat([pred_pd, targ_pd], axis=1, join_axes=[pred_pd.index])
	final_pd = combined_pd.drop(combined_pd[(combined_pd.targ_nonattack == 0.0) & (combined_pd.targ_attack == 0.0)].index)
	pred = final_pd.iloc[:, :2].copy()
	targ = final_pd.iloc[:, 2:].copy()
	_test_precision,_test_recall,_test_f1,_support = precision_recall_fscore_support(targ, pred)
	#import pdb
	#pdb.set_trace()	
	
	pred_attack = pred.iloc[:, 1].copy()
	targ_attack = targ.iloc[:, 1].copy()

	acc = accuracy_score(targ_attack, pred_attack)
	conf = confusion_matrix(targ_attack, pred_attack)
	#import pdb
	#pdb.set_trace()	
	#print "threshold: ", threshold
	#print conf
	#print "\n"
	tn, fp, fn, tp = conf.ravel()

	fpr = float(fp) / (fp + tn + 1e-10)
	fnr = float(fn)/ (fn + tp + 1e-10)

	far = (fpr + fnr) / 2
	#print conf
	return _test_precision, _test_recall, _test_f1, _support, acc, far 

def shuffle_data(data, label, pad_idx):
    
	idx = np.arange(len(data)) #create array of index for data
	np.random.shuffle(idx) #randomly shuffle idx
	
	data = data[idx]
	label = label[idx]
	pad_idx = pad_idx[idx]
	
	return data, label,pad_idx

def shuffle_fixed_data(data, label):
    
	idx = np.arange(len(data)) #create array of index for data
	np.random.shuffle(idx) #randomly shuffle idx
	data = data[idx]
	label = label[idx]
	
	return data, label

def plot_precision(test_pred, test_targ, title):

	labels = ['precision', 'recall', 'accuracy', 'FAR']	
	precision = []
	recall = []
	accuracy = []
	FAR = []

	thresholds = np.arange(0.999955, 1., 0.0000001)
	for threshold in thresholds:
		_test_precision,_test_recall,_test_f1,_support,acc,far = get_result(test_pred, test_targ, threshold)
		
		precision.append(_test_precision[1])
		recall.append(_test_recall[1])
		accuracy.append(acc)
		FAR.append(far)

	plot_data = dict()
	plot_data["precision"] = precision 
	plot_data["recall"] = recall
	plot_data["accuracy"] = accuracy 
	plot_data["FAR"] = FAR 

	#import pdb
	#pdb.set_trace()
	fig, ax = plt.subplots(figsize = (10,8))		
	for label in labels:
		ax.plot(thresholds, np.array(plot_data[label]), label=label)	

	ax.legend()
	plt.title(title)
	plt.xlabel('threshold')
	plt.show(block=False)
	fig.savefig('unsw.png')
	raw_input("Press Enter to Exit")

def plot_training(val_loss, val_acc, val_far, train_loss, train_acc, title):

	labels = ['val_loss', 'val_acc', 'val_far', 'train_loss', 'train_acc']	

	plot_data = dict()
	plot_data["val_loss"] = val_loss 
	plot_data["val_acc"] = val_acc
	plot_data["val_far"] = val_far 
	plot_data["train_loss"] = train_loss 
	plot_data["train_acc"] = train_acc 

	#import pdb
	#pdb.set_trace()
	fig, ax = plt.subplots(figsize = (10,8))		
	for label in labels:
		ax.plot(np.arange(len(plot_data[label])), np.array(plot_data[label]), label=label)	

	ax.legend()
	plt.title(title)
	plt.xlabel('no. of epochs')
	plt.show(block=False)
	raw_input("Press Enter to Exit")

