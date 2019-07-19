import numpy as np
import gc
import pdb

def create_windows(fileName, window_size, overlap):
	print( "Reading data")
	origData = fileName 
	startIndex=0
	endIndex = window_size
	windows_list = []
	totalSamples = origData.shape[0]
	print( "Creating Windows")
	while endIndex <= totalSamples:
		window = origData[startIndex:endIndex]
		windows_list.append(window)
		startIndex += window_size - overlap
		endIndex = startIndex + window_size
	del origData
	gc.collect()
	print( "Creating a numpy ndarray")
	np_windowed = np.array(windows_list) #create 3d array
	#pdb.set_trace()
	del windows_list
	gc.collect()
	
	

	return np_windowed

def create_windows_raw_label(fileName, window_size, overlap):
	print( "Reading data")
	origData = fileName 
	#pdb.set_trace()
	startIndex=0
	endIndex = window_size
	windows_list = []
	totalSamples = origData.shape[0]
	print( "Creating Windows")
	while endIndex <= totalSamples:
		window = origData[startIndex:endIndex]
		windows_list.append(window)
		startIndex += window_size - overlap
		endIndex = startIndex + window_size
	del origData
	gc.collect()
	print( "Creating a numpy ndarray")
	np_windowed = np.array(windows_list)
	del windows_list
	gc.collect()

	np_windowed = np_windowed.reshape(np_windowed.shape[0],np_windowed.shape[1],  1) #convert to 3d array

	return np_windowed

