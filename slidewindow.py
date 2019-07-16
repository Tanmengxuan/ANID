import numpy as np
import os.path
import argparse
import gc
import csv
import yaml
import pdb

parser = argparse.ArgumentParser()


parser.add_argument('--path', type = str, help = 'path of file to preprocess')
parser.add_argument('--window', type = int, help = 'Number of samples in one window')
parser.add_argument('--overlap', type = int, help = 'Number of samples overlapping between subsequent windows')
parser.add_argument('--name', type = str, help = 'name of output file')

parser.add_argument('--raw_label', action = 'store_true', help = 'process raw_label')


args = parser.parse_args()



#def create_windows(fileName, window_size, overlap, prefix):
def create_windows(fileName, window_size, overlap):
	print( "Reading data")
	#origData = np.loadtxt(fileName,delimiter='\t', dtype =np.float32)
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
	np_windowed = np.array(windows_list)
	del windows_list
	gc.collect()
	
	#pdb.set_trace()
	#np_windowed = np_windowed.reshape(-1, np_windowed.shape[2])
	#np_windowed = np_windowed.reshape(-1, 1)
	#save_path = os.path.join(os.path.dirname(fileName),'normed_allreducedselect'+'_w'+str(window_size)+'o'+str(overlap)+ '_' + prefix + ".csv")
	#np.savetxt(save_path,np_windowed,delimiter='\t',fmt='%f')
	#np.savetxt(save_path,np_windowed,delimiter='\t',fmt='%s')
	#print ('saved to: ', save_path) 
	return np_windowed

#def create_windows_raw_label(fileName, window_size, overlap, prefix):
def create_windows_raw_label(fileName, window_size, overlap):
	print( "Reading data")
	#origData = np.loadtxt(fileName,delimiter='\t', dtype =str)
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
	#save_path = os.path.join(os.path.dirname(fileName),'normed_allreducedselect_labels'+'_w'+str(window_size)+'o'+str(overlap)+ '_' + prefix + ".yml")
	#fp = yaml.dump({'label':np_windowed})
	#open(save_path, 'w').write(fp)
	#print ('saved to: ', save_path) 

	return np_windowed


#if __name__ == '__main__':
#
#	if args.raw_label:
#		
#		create_windows_raw_label(args.path, args.window, args.overlap, args.name)
#	else:
#
#		create_windows(args.path, args.window, args.overlap, args.name)
	

