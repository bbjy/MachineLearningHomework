import os
import numpy as np
import time
import processdata
import Adaboost

from sklearn.model_selection import train_test_split
from sklearn.datasets.base import Bunch
from sklearn.tree import DecisionTreeClassifier

def test_dota2():
	path_dota2 = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/Dota2-Games-Dataset/"
	train_dota2 = os.path.join(path_dota2,"dota2Train.csv")
	test_dota2 = os.path.join(path_dota2,"dota2Test.csv")

	x_train,y_train = processdata.process_dota2(train_dota2)
	x_test,y_test = processdata.process_dota2(test_dota2)

	model = Adaboost.Adaboost(x_train, y_train,estimator = DecisionTreeClassifier, itern = 1000,max_depth = 3)
	print "Start training Adaboost.MR model on dota2!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	print "Training end!"
	train_score = model.score(x_train,y_train)
	print "Train score: " ,train_score
	print "Train time: " ,train_time,"s"
	print "Start testing! " 
	testscore = model.score(x_test,y_test)
	print "test score: " ,testscore
	print "\n"

# def test_chess():
# 	chess_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/chess"
# 	chess_data = os.path.join(chess_path,"krkopt.data")

# 	X, y = processdata.process_chess(chess_data)

# 	# splite dataset to traindataset and test dataset :
# 	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2)

# 	model = Adaboost.AdaboostMR(train_data, train_target,18, estimator = DecisionTreeClassifier, itern = 100)
# 	print "Start training Adaboost.MR model on chess!"
# 	time_start = time.time()
# 	model.train()
# 	time_end = time.time()
# 	train_time = time_end - time_start
# 	print "Training end!"
# 	train_correct,train_total,train_score = model.test(train_data,train_target)
# 	print "Correct data: " ,train_correct, "Total data: ",train_total
# 	print "Train score: %.4f" %(train_score)
# 	print "Train time: " ,train_time,"s"
	# print "Start testing! " 
	# # correct,total,score = model.test(test_data,test_target)
	# test_correct,test_total,test_score = model.test(test_data,test_target)
	# print "Correct data: " ,test_correct, "Total data: ",test_total
 	#print "Test score: %.4f" %(test_score)
	# print "test score: " ,testscore
	# print "\n"


def test_letter():
	letter_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/letter-recognition"
	letter_data = os.path.join(letter_path,"letter-recognition.data")

	X, y = processdata.process_letter(letter_data)

	# splite dataset to traindataset and test dataset :
	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)
	model = Adaboost.AdaboostMR(train_data, train_target,26, estimator = DecisionTreeClassifier, itern = 1000,max_depth = 3)
	print "Start training Adaboost.MR model on letter!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	print "Training end!"
	train_correct,train_total,train_score = model.test(train_data,train_target)
	print "Correct data: " ,train_correct, "Total data: ",train_total
	print "Train score: %.4f" %(train_score)
	print "Train time: " ,train_time,"s"

	print "Start testing! " 
	# correct,total,score = model.test(test_data,test_target)
	test_correct,test_total,test_score = model.test(test_data,test_target)
	print "Test score: %.4f" %(test_score)
	print "\n"


def test_car():
	car_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/car"
	car_data = os.path.join(car_path,"car.data")

	X, y = processdata.process_car(car_data)

	# splite dataset to traindataset and test dataset :
	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)

	model = Adaboost.AdaboostMR(train_data, train_target,4, estimator = DecisionTreeClassifier, itern = 1000,max_depth = 3)
	print "Start training Adaboost.MR model on car!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	print "Training end!"
	train_correct,train_total,train_score = model.test(train_data,train_target)
	print "Correct data: " ,train_correct, "Total data: ",train_total
	print "Train score: %.4f" %(train_score)
	print "Train time: " ,train_time,"s"

	print "Start testing! " 
	test_correct,test_total,test_score = model.test(test_data,test_target)
	print "Correct data: " ,test_correct, "Total data: ",test_total
	print "Test score: %.4f" %(test_score)
	print "\n"


# def test_Abalone():
# 	abalone_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/Abalone"
# 	abalone_data = os.path.join(abalone_path,"abalone.data")

# 	X, y = processdata.process_abalone(abalone_data)

# 	# splite dataset to traindataset and test dataset :
# 	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)

# 	model = Adaboost.AdaboostMR(train_data, train_target,28, estimator = DecisionTreeClassifier, itern = 100)
# 	print "Start training Adaboost model on Abalone!"
# 	time_start = time.time()
# 	model.train()
# 	time_end = time.time()
# 	train_time = time_end - time_start
# 	model.test(train_data,train_target)
# 	print "Train time: " ,train_time,"s"
# 	train_correct,train_total,train_score = model.test(train_data,train_target)
# 	print "Correct data: " ,train_correct, "Total data: ",train_total
# 	print "Train score: %.4f" %(train_score)
# 	print "Train time: " ,train_time,"s"

# 	# print "Start testing! " 
	# # correct,total,score = model.test(test_data,test_target)
	# test_correct,test_total,test_score = model.test(test_data,test_target)
	# print "Correct data: " ,test_correct, "Total data: ",test_total
 #    print "Test score: %.4f" %(test_score)
	print "\n"

def test_iris():
	iris_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/iris"
	iris_data = os.path.join(iris_path,"Iris.data")

	X, y = processdata.process_iris(iris_data)

	# splite dataset to traindataset and test dataset :
	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)

	model = Adaboost.AdaboostMR(train_data, train_target,3, estimator = DecisionTreeClassifier, itern = 1000,max_depth = 3)
	print "Start training Adaboost model on Iris!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	print "Training end!"
	train_correct,train_total,train_score = model.test(train_data,train_target)
	print "Correct data: " ,train_correct, "Total data: ",train_total
	print "Train score: %.4f" %(train_score)
	print "Train time: " ,train_time,"s"

	print "Start testing! " 
	# correct,total,score = model.test(test_data,test_target)
	test_correct,test_total,test_score = model.test(test_data,test_target)
	print "Correct data: " ,test_correct, "Total data: ",test_total
	print "Test score: %.4f" %(test_score)
	print "\n"

def test_banknote():
	banknote_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/banknote"
	banknote_data = os.path.join(banknote_path,"data_banknote_authentication.txt")

	X, y = processdata.process_banknote(banknote_data)

	# splite dataset to traindataset and test dataset :
	train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2,random_state=3)

	model = Adaboost.AdaboostMR(train_data, train_target,2, estimator = DecisionTreeClassifier, itern = 1000,max_depth = 3)
	print "Start training Adaboost model on Banknote!"
	time_start = time.time()
	model.train()
	time_end = time.time()
	train_time = time_end - time_start
	print "Training end!"
	train_correct,train_total,train_score = model.test(train_data,train_target)
	print "Correct data: " ,train_correct, "Total data: ",train_total
	print "Train score: %.4f" %(train_score)
	print "Train time: " ,train_time,"s"

	print "Start testing! " 
	# correct,total,score = model.test(test_data,test_target)
	test_correct,test_total,test_score = model.test(test_data,test_target)
	print "Correct data: " ,test_correct, "Total data: ",test_total
	print "Test score: %.4f" %(test_score)
	print "\n"

if __name__ == '__main__':

	
	test_dota2()

	# test_chess()

	test_letter()

	test_car()

	# test_Abalone()

	test_iris()

	test_banknote()
