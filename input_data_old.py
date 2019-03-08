import glob
import csv
import re
import numpy as np
import pandas as pd
from collections import Counter
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle

class inputter:
	
	def __init__(self,filepathGlob, maxlen, pad_value, num_split):
		files = glob.glob(filepathGlob)
		#pdb.set_trace()
		self.trainFile = ''
		self.validationFile = ''
		self.testFile = ''
		self.windowSize = None
		self.overlap = None
		self.maxlen = maxlen
		self.pad_value = pad_value
		self.num_split = num_split
		for file in files:
			if re.search(r'test\.csv',file):
				self.testFile = file
				print "Test file read: "+file
				self.extractWindowOverlap(file)
			elif re.search(r'train\.csv',file):
				self.trainFile = file
				print "Train file read: " + file
				self.extractWindowOverlap(file)
			elif re.search(r'validation\.csv',file):
				self.validationFile = file
				print "Validation file read: "+file
				self.extractWindowOverlap(file)
		#assert(self.trainFile), "Training file not found, check Glob passed"
		#assert(self.validationFile)	, "Validation file not found, check Glob passed"
		#assert(self.testFile), "Test File not found, check Glob passed"

	def extractWindowOverlap(self,file):
		m = re.search(r'_w(\d+)o(\d+)_',file)
		assert(m), "Couldn't find window size and overlap size in file " + file
		if self.windowSize is None:
			self.windowSize = int(m.group(1))
			self.overlap = int(m.group(2))
		else:
			assert (int(m.group(1)) == self.windowSize and int(m.group(2)) == self.overlap),"Window or overlap size not same for all files"
	
	def determineTrainWindows(self):
		if not self.trainFile:
			raise ValueError("No training file has been read, please specify")
		with open(self.trainFile,"r") as trainFile:
			num_lines = sum(1 for line in trainFile)
		return num_lines/self.windowSize
	

	def trainingBatches(self, batchSize):
		if not self.trainFile:
			raise ValueError("No training file has been read, please specify")
		noloop = 0
		return self.generateBatches(self.trainFile, batchSize, noloop )

	def validationBatches(self,batchSize):
		if not self.validationFile:
			raise ValueError("No validation file has been read, please specify")
		loop= 1
		return self.generateBatches(self.validationFile, batchSize, loop)

	def generateBatches(self, filePath, batchSize,loop):
		while True : #for looping over the same file
			batchData = []
			batchLabels = []
			windowLabels = []
			windowData = []
			windowLocation = 0
			batchLocation = 0

			with open(filePath, "r") as trainFile:

				trainCsv = csv.reader(trainFile, delimiter = '\t')
				for row in trainCsv:

					if windowLocation < self.windowSize:
						data = np.array(list(map(np.float32 , row[1:])))
						label = int(float(row[0]))
						windowData.append(data)
						windowLabels.append(label)

						windowLocation +=1
					if windowLocation == self.windowSize:
						windowLocation = 0
						windownp = np.array(windowData)
						labelsnp = np.array(windowLabels)
						batchData.append(windownp)
						one_hot = np.zeros(2,dtype=np.int8)
						one_hot[int(any(labelsnp))] = 1
						batchLabels.append(one_hot)
						del windowData[:]
						del windowLabels[:]
						batchLocation +=1
					if batchLocation == batchSize:
						batchLocation = 0
						batchDatanp = np.array(batchData)
						batchLabelsnp = np.array(batchLabels)

						shapeData = batchDatanp.shape
						batchDatanp = batchDatanp.reshape(shapeData[0],shapeData[1],shapeData[2],1)
						del batchData[:]
						del batchLabels[:]
						yield [batchDatanp, batchLabelsnp]
			if not loop:
				break

	def getValidationData(self, mergeAttackLabelsWithAtleast=False):
		if not self.validationFile:
			raise ValueError("No Validation file has been read, please specify")
		return inputWindowedData(self.validationFile,self.windowSize, mergeAttackLabelsWithAtleast)

	def getTestData(self,mergeAttackLabelsWithAtleast=False):
		if not self.testFile:
			raise ValueError("No test file has been read, please specify")
		return inputWindowedData(self.testFile, self.windowSize,mergeAttackLabelsWithAtleast)

	def getTrainData(self,mergeAttackLabelsWithAtleast=False):
		if not self.trainFile:
			raise ValueError("No train file has been read, please specify")
		return inputWindowedData(self.trainFile, self.windowSize, mergeAttackLabelsWithAtleast)

	def getPadValidationData(self):
		if not self.validationFile:
			raise ValueError("No Validation file has been read, please specify")
		return get_padded(self.validationFile,self.maxlen, self.pad_value, self.num_split)

	def getPadTestData(self):
		if not self.testFile:
			raise ValueError("No test file has been read, please specify")
		return get_padded(self.testFile, self.maxlen, self.pad_value, self.num_split)

	def getPadTrainData(self):
		if not self.trainFile:
			raise ValueError("No train file has been read, please specify")
		return get_padded(self.trainFile, self.maxlen, self.pad_value, self.num_split)

def inputWindowedData(filePath,windowSize,mergeAttackLabelsWithAtleast=False):
	#windowSize = 20
	with open(filePath,"r") as file:
		data=[]
		labels=[]
		windowLocation = 0
		windowData = []
		windowLabels = []
		#validCsv = csv.reader(file, delimiter = ',') #for cicids data before normalization
		validCsv = csv.reader(file, delimiter = '\t')
		for row in validCsv:
			if windowLocation < windowSize:
				datarow = np.array(map(np.float32 , row[1:]))
				#import pdb
				#pdb.set_trace()
				labelrow = int(float(row[0]))
				windowData.append(datarow)
				windowLabels.append(labelrow)
				windowLocation +=1
			if windowLocation == windowSize:
				windowLocation = 0
				windownp = np.array(windowData) #windownp.shape = (windowSize, featuresize)
				labelsnp = np.array(windowLabels) #labelsnp.shape = (windowSize,)
				data.append(windownp)
				if mergeAttackLabelsWithAtleast == False: #presevers all original labels at each time stp 
					one_hot = np.zeros((labelsnp.shape[0],2), dtype = np.int8)
					for i, labelrow in enumerate(labelsnp):
						one_hot[i][int(labelsnp[i])] = 1 # one_hot.shape = (windowSize, 2)
				else:
					one_hot = np.zeros(2,dtype=np.int8) #looks at all labels across one window and determines final label according to given condition
					cnt = Counter(labelsnp)
					one_hot[int(cnt[1] >= mergeAttackLabelsWithAtleast)] = 1 # one_hot.shape = (2,)
				labels.append(one_hot)
				del windowData[:]
				del windowLabels[:]
	datanp = np.array(data)
	labelsnp = np.array(labels)
	shapeData = datanp.shape
	datanp = datanp.reshape(shapeData[0],shapeData[1],shapeData[2],1)
	return [datanp,labelsnp]


def getMaster(labels, num_split):

	master_list = []
	attack_list = []
	for i in range(len(labels)):
	
		if i != (len(labels) - 1):
		
			if labels[i] == 0:
				attack_list.append(i)
			
			elif labels[i] ==1 and labels[i + 1] == 1:
				attack_list.append(i)
			
			elif labels[i] ==1 and labels[i + 1] == 0:
				attack_list.append(i)
				attack_list = np.array_split(attack_list,num_split)
				master_list.append(attack_list)
				attack_list = []

		else:
		
			if labels[i] == 0:
				attack_list.append(i)
				attack_list = np.array_split(attack_list,num_split)
				master_list.append(attack_list)
			
			elif labels[i] == 1:
				attack_list.append(i)
				attack_list = np.array_split(attack_list,num_split)
				master_list.append(attack_list)
	
	return master_list
	
#fast implementation version
def check_pad_condition(data, pad_value):
	count = 0
	for row in range(len(data)):
		for element in data.iloc[row,:]:
			if element == pad_value:
				count += 1
                
	print "Number of {} in data: {}".format(pad_value, count)

def get_pad_idx(label_unpad, Maxlen):

	pad_idx = []

	for sample in range(len(label_unpad)):
		pad_idx.append(len(label_unpad[sample]))

	pad_idx = np.array(pad_idx)
	#pad_idx = Maxlen - pad_idx
	return pad_idx

def get_padded(file_path, Maxlen, pad_value, num_split):
    
	ValidPath = file_path
	ValidCsv = pd.read_csv(ValidPath ,sep="\t", index_col=None, header=None, dtype='float64')
	ValidLabel = ValidCsv.iloc[:,[0]].copy()
	ValidData = ValidCsv.iloc[:,1:].copy()
	#ValidData.drop(ValidData.columns[[493,669,844, 868]], axis=1, inplace=True)
	#check_pad_condition(ValidData, pad_value)
	del ValidCsv
	valid_idx = getMaster(ValidLabel[0].tolist(), num_split)
	
	data = []
	label = []
	
	for window in valid_idx:
		for subwindow in window:
			subwindow_data =  ValidData[ValidData.index.isin(subwindow)].copy()
			
			subwindow_label = ValidLabel[ValidLabel.index.isin(subwindow)].copy()
			subwindow_label_onehot = to_categorical(subwindow_label, num_classes = 2)
			
			data.append(subwindow_data)
			label.append(subwindow_label_onehot)
	
	pad_idx = get_pad_idx(label, Maxlen)
	pad_validdata = pad_sequences(data, padding = "post", maxlen = Maxlen, value =pad_value, dtype='float32')
	pad_validlabel = pad_sequences(label, padding = "post", maxlen = Maxlen, value =0.0, dtype='float32')
	pad_validdata = np.round(pad_validdata, 5)		
	return pad_validdata, pad_validlabel, pad_idx

def create_mask(padded_idx, MAXLEN):
    
    mask = np.ones((len(padded_idx),MAXLEN,MAXLEN))
    
    for i in range(len(padded_idx)):
        
        mask[i, :, padded_idx[i]: ] = 0
        mask[i, padded_idx[i]:, 1: ] = 0 
        
    return mask

class Get_data(object):

	def __init__(self, input_path, maxlen, pad_value, window_split):

		self.input_path = input_path 
		self.maxlen = maxlen
		self.pad_value = pad_value
		self.window_split = window_split
		self.inp = inputter(self.input_path, self.maxlen, self.pad_value, self.window_split) 

	def get_input(self, data_type):

		method_name = 'getPad' + data_type + 'Data' 
		inputX, inputY, input_pad_idx = getattr(self.inp, method_name)()	

		inputX = inputX.reshape(inputX.shape[0],inputX.shape[1],inputX.shape[2])
		input_mask = create_mask(input_pad_idx, self.maxlen)
	
		print data_type + " data shape", inputX.shape, inputY.shape, input_pad_idx.shape
		print data_type + " mask", input_mask.shape

		return inputX, inputY, input_pad_idx, input_mask

class Fixed_seq(Get_data):

	def get_input(self, data_type, SlidingAttackLabel = False):

		method_name = 'get' + data_type + 'Data' 
		inputX, inputY = getattr(self.inp, method_name)(mergeAttackLabelsWithAtleast=SlidingAttackLabel)	
		inputX = inputX.reshape(inputX.shape[0],inputX.shape[1],inputX.shape[2])

		print data_type + " data shape", inputX.shape, inputY.shape

		return inputX, inputY

	def get_pickle(self, data_type):

		file_path = getattr(self.inp, data_type + 'File')
		with open(file_path, 'rb') as file_handle:
			data = pickle.load(file_handle)

		inputX = data[ data_type + '_x']	
		inputY = data[ data_type + '_y']	

		print data_type + " data shape", inputX.shape, inputY.shape

		return inputX, inputY



