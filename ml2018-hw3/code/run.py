import os
import numpy as np
import time
import processdata
import Multiboosting
import multi_boosting
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import Bunch
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def test_krkopt(model,itern=10):
	chess_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework3/data"
	chess_data = os.path.join(chess_path,"krkopt.data")
	X, y = processdata.process_krkopt(chess_data)

	# splite dataset to traindataset and test dataset :
	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)
	if model == 'Adaboost':
		model = Multiboosting.Adaboost(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Adaboost model on chess!"
	elif model=='Multiboosting':
		model = Multiboosting.MultiBoosting(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Multiboosting model on chess!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	# print "Training end!"
	# print type(test_data), type(test_target)
	final_result,test_score = model.test(test_data,test_target)
	return test_target,final_result,test_score 

'''
def test_krkopt(model,itern=10):
	# For test vias and variance but the result is not good
	chess_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework3/data"
	chess_data = os.path.join(chess_path,"krkopt.data")
	x,y = processdata.process_krkopt(chess_data)
	all_result = np.ndarray((x.shape[0]))
	kf = KFold(n_splits=3)
	sum_score = 0.0
	for train_index, test_index in kf.split(x):
		train_data, test_data = x[train_index], x[test_index]
		train_target,test_target = y[train_index],y[test_index]


	# train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.2,random_state=3)

	# splite dataset to traindataset and test dataset :
	# train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2)
		if model == 'Adaboost':
			model = Multiboosting.Adaboost(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Adaboost model on chess!"
		elif model=='Multiboosting':
			model = Multiboosting.MultiBoosting(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Multiboosting model on chess!"
		time_start = time.time()
		model.train()
		time_end = time.time()
		train_time = time_end - time_start
		# print "Training end!"
		# print type(test_data), type(test_target)
		final_result,test_score = model.test(test_data,test_target)
		# print "test_score in test_krkop",test_score
		all_result[test_index] = final_result
		sum_score += test_score
	mean_score = sum_score*1.0 / 3

	return y,all_result,mean_score
'''

def test_letter(model,itern=10):
	letter_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework3/data"
	letter_data = os.path.join(letter_path,"letter-recognition.data")

	X, y = processdata.process_letter(letter_data)

	# splite dataset to traindataset and test dataset :
	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)
	if model == 'Adaboost':
		model = Multiboosting.Adaboost(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Adaboost model on chess!"
	elif model=='Multiboosting':
		model = Multiboosting.MultiBoosting(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Multiboosting model on chess!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	# print "Training end!"
	# print type(test_data), type(test_target)
	final_result,test_score = model.test(test_data,test_target)
	return test_target,final_result,test_score


def test_iris(model,itern=10):
	iris_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework3/data"
	iris_data = os.path.join(iris_path,"Iris.data")

	X, y = processdata.process_iris(iris_data)

	# splite dataset to traindataset and test dataset :
	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)

	if model == 'Adaboost':
		model = Multiboosting.Adaboost(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Adaboost model on chess!"
	elif model=='Multiboosting':
		model = Multiboosting.MultiBoosting(train_data, train_target,estimator = DecisionTreeClassifier, itern = itern)
		# print "Start training Multiboosting model on chess!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	# print "Training end!"
	# print type(test_data), type(test_target)
	final_result,test_score = model.test(test_data,test_target)
	return test_target,final_result,test_score

def bias_and_var(final_result,y):
	
	# mean_score  = 0.0
	bias = []  
	#the proportion of classifications that are both incorrect and equal to the central tendency.
	variance = []
	 #the proportion of classifications that are both incorrect and not equal to the central tendency

	final_result = np.asarray(final_result, dtype=int)
	# print type(final_result[0][0])
	for i in range(y.shape[0]):
		counts = np.bincount(final_result[:,i])
		central_tendency = np.argmax(counts)
		bia = 0.0
		var = 0.0
		for t in range(final_result.shape[0]):
			if final_result[t,i] != y[i] and final_result[t,i] == central_tendency:
				bia += 1
			if final_result[t,i] != y[i] and final_result[t,i] != central_tendency:
				var += 1
		
		bias.append(bia)
		variance.append(var)
	# print "bias",bias
	bias = [b * 1.0 / final_result.shape[0] for b in bias]
	variance = [v * 1.0 / final_result.shape[0] for v in variance]
	mean_bias = sum(bias) / len(bias)
	mean_var = sum(variance) / len(variance)
	# print "mean_bias: ",mean_bias, "mean_var: ",mean_var
	return mean_bias,mean_var

if __name__ == '__main__':

	# print "test_score of Adaboost on krkopt: (ronud:10)"
	'''	
    #For test bias and variance
	sum_score = 0.0
	chess_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework3/data"
	chess_data = os.path.join(chess_path,"krkopt.data")


	x,y = processdata.process_krkopt(chess_data)
	bias = np.ndarray((10,y.shape[0]))
	variance = np.ndarray((10,y.shape[0]))
	predict_y = np.ndarray((10,y.shape[0]))

	for i in range(10):
		#y,all_result,mean_score
		test_target,final_result,test_score = test_krkopt('Adaboost',10)
		# print "test_score",test_score
		# print "\n"
		sum_score +=test_score
		
		# print final_result.shape
		predict_y[i] = np.array(final_result)

	mean_bias,mean_var = bias_and_var(predict_y,y)

	# mean_bias = sum(bias) / len(bias)
	# mean_var = sum(variance) / len(variance)
	mean_score = sum_score / 10.0
	# mean_bias,mean_var = bias_and_var(test_krkopt,'Adaboost',10)

	print "test_score of Adaboost on krkopt:(ronud:10) "
	# test_target,final_result,test_score = test_krkopt('Multiboosting',10)
	print mean_score
	print "\n"

	print "bias_and_var of Adaboost on krkopt:(ronud:10) "
	print "mean_bias: ", mean_bias, "\nmean_varance: ",mean_var
	print "\n"

	sum_score = 0.0
	for i in range(10):
		#y,all_result,mean_score
		test_target,final_result,test_score = test_krkopt('Multiboosting',10)
		# print "test_score",test_score
		# print "\n"
		sum_score +=test_score
		
		# print final_result.shape
		predict_y[i] = np.array(final_result)

	mean_bias,mean_var = bias_and_var(predict_y,y)

	# mean_bias = sum(bias) / len(bias)
	# mean_var = sum(variance) / len(variance)
	mean_score = sum_score / 10.0
	# mean_bias,mean_var = bias_and_var(test_krkopt,'Adaboost',10)

	print "test_score of Multiboosting on krkopt:(ronud:10) "
	# test_target,final_result,test_score = test_krkopt('Multiboosting',10)
	print mean_score
	print "\n"
	
	print "bias_and_var of Adaboost on krkopt:(ronud:10) "
	print "mean_bias: ", mean_bias, "\nmean_varance: ",mean_var
	print "\n"
	'''
	#-----------------------------------------------------------------------
	print "test_score of Adaboost on krkopt: (ronud:10)"
	# test_score = test_krkopt('Adaboost',10)
	test_target,final_result,test_score = test_krkopt('Adaboost',10)
	print test_score
	print "\n"

	print "test_score of Multiboosting on krkopt:(ronud:10) "
	# test_krkopt('Multiboosting',10)
	test_target,final_result,test_score = test_krkopt('Multiboosting',10)
	print test_score
	print "\n"
	#-----------------------------------------------------------------------
	print "test_score of Adaboost on iris: (ronud:10)"
	test_target,final_result,test_score = test_iris('Adaboost',10)
	print test_score
	print "\n"

	print "test_score of Multiboosting on iris:(ronud:10) "
	test_target,final_result,test_score = test_iris('Multiboosting',10)
	print test_score
	print "\n"

	#---------------------------------------------------------------------

	print "test_score of Adaboost on letter: (ronud:10)"
	test_target,final_result,test_score = test_letter('Adaboost',10)
	print test_score
	print "\n"

	
	print "test_score of Multiboosting on letter:(ronud:10) "
	test_target,final_result,test_score = test_letter('Multiboosting',10)
	print test_score
	print "\n"

