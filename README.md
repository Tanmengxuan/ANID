# cicids2017

Shape of data : (num of samples, window length, feature size)
Shape of label: (num of samples, window length, num of class)

num of class = 2

flags:

--train : train the model
--num_feature : feature size
--MAXLEN : window length
--drop: the amount of probability to keep in dropout ( from 0.0 to 1.0 )
--num_hidden: size of network
--num_epoch: number of epochs for training
--atten: whether or not to use attention
--model_name: name of saved model
--test: test the saved model specify in --model_name

example command to run cicids.py
python cicids.py --train --num_features 19 --MAXLEN 10 --drop 0.9 --num_hidden 100 --num_epoch 50 --atten --model_name cicids25 --test
