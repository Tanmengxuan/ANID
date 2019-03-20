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

# Preprocess data

1) python feature_select.py

2) python L2_normalization.py --path allreducedselect_w10o0_\*

3) python slidewindow.py --path normed_allreducedselect_w10o0_train.csv  --window 10 --overlap 9 --name train

 python slidewindow.py --path normed_allreducedselect_w10o0_validation.csv  --window 10 --overlap 9 --name validation

 python slidewindow.py --path normed_allreducedselect_w10o0_test.csv  --window 10 --overlap 9 --name test

4) python shuffle_data.py
