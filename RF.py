import numpy as np
import sklearn as sk
from numpy import genfromtxt
import pandas as pd 
from sklearn import tree as dt
import random
import math
import matplotlib.pyplot as plt
from scipy import stats as stat
from sklearn.metrics import confusion_matrix

def read_dat(dataset):
	#This method reads the dataset into pandas dataframes and inserts column names for ease of access.
	
	#WINE
	if(dataset=='wine'):
		df=pd.read_csv('wine.txt')
		df.columns=['label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoids phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315','Proline']
		rownum=df.shape[0]
		sub=rownum/4
		sub1=sub*3
		trainrows=random.sample(range(0,rownum),sub)
		train=df.iloc[trainrows,:]
		test=df.copy()
		test=test.drop(test.index[trainrows])

	#MNIST
	if(dataset== 'mnist'):
		train=pd.read_csv('train.csv')
		test=pd.read_csv('test.csv')
		train=train.transpose()						# The MNIST data must be transposed before processing
		test=test.transpose()

		# Adding column names to MNIST Training and Test DataFrame 
		name=[]
		for i in range(train.shape[1]-1):
			name.append(str(i))
		name.append('label')
		train.columns= name
		test.columns=name
	
	return train,test

def test_forest(rf,test):

	#In the function trees, the training and testing dataframes are provided. 
	#1. Labels are extracted from the dataframes.
	#2. Decision Tree is trained on training data and labels y
	#3. Testing data is provided to the tree for prediction
	#4. Predicted y and actual y are compared to measure correctness of classification	
	test_results=np.ndarray(shape=(test.shape[0],len(rf)))
	ytest=np.asarray(list(test.label))
	ytest=np.reshape(ytest,(test.shape[0],-1))
	test=test.drop('label',1)
	test_arr=np.asarray(test)
	for i in range(len(rf)):
	#Testing and Measurement of Correctness
		for j in range(len(test_arr)):
			row=np.reshape(test_arr[j],(1,test.shape[1]))
			test_results[j,i]=rf[i].predict(row)
	ypred=[]
	for row in test_results:

		temp=str(stat.mode(row)[0])
		temp=temp.split(".")[0]
		temp=temp.split(" ")[1]
		ypred.append(int(temp))

	return ypred,ytest

def error(ypred,ytest,flag):
	count=0
	for i,l in zip(range(len(ypred)),ytest):
		if ypred[i]==l:
			count+=1
	correct=1- (float(count)/float(len(ypred)))
	if flag==0:
		print confusion_matrix(ytest,ypred);
	return correct


def outofbag(tree,oob,hasht):
	print "OOB"
	ytest=np.asarray(list(oob.label))
	ytest=np.reshape(ytest,(oob.shape[0],-1))
	oob=oob.drop('label',1)
	test_arr=np.asarray(oob)
	data_ix=oob.index.values;

	for i in range(len(test_arr)):
			row=np.reshape(test_arr[i],(1,oob.shape[1]))
			prediction=tree.predict(row)
			if(data_ix[i] in hasht.keys()):
				hasht[data_ix[i]].append(prediction)
			else:
				hasht[data_ix[i]]=[]
				hasht[data_ix[i]].append(prediction)
	return hasht

def oob_Error(train,hasht):
	print "OOB ERROR"
	ind=train.index.values
	ypred=[]
	labels=[]
	for index in ind:
		if(index in hasht.keys()):
			row=hasht[index]
			temp=stat.mode(row)[0]
			ypred.append(int(temp[0]))
			labels.append(int(train.loc[index,'label']))
		else:
			pass
	return error(ypred,labels,1)

def train_forest(train,b):
	print "TRAIN"
	#1. DECLARE a list of trees
	#2. CREATE b number of trees using in a loop.
	#3. For each tree, select 66% of the total training set 
	train_feat=len(train.columns)-1
	# print int(math.ceil(math.sqrt(train_feat)))
	train_ind=train.index.values
	r_forest=[]
	temp=dict()
	# keys=train.index.values
	# for val in keys:
	# 	temp[val]=[]
	for i in range(0,b):
			#Decision Tree object Declaration
		tr=dt.DecisionTreeClassifier(max_depth=5,max_features=int(math.ceil(math.sqrt(train_feat))))
		rownum=train.shape[0]
		sub=rownum/5
		trainrows=list(np.random.choice(range(0,rownum),sub))
		# trainrows=random.sample(range(0,rownum),sub)
		# print "SAMPLE"
		# print trainrows		
		train1=train.iloc[trainrows,:]
		y=np.asarray(list(train1.label))
		y=np.reshape(y,(train1.shape[0],-1))
		train1=train1.drop('label',1)
		train1_arr=np.asarray(train1)
		#Training
		tr=tr.fit(train1_arr,y)
		oob=train.copy()
		oob=oob.drop(oob.index[trainrows])
		temp=outofbag(tr,oob,temp)
		r_forest.append(tr)

	ooberror=oob_Error(train,temp)

	return r_forest,ooberror

	#MAIN
if __name__=="__main__":


	correct=[]
	oobs=[]
	train,test=read_dat('wine')    #Change to mnist for mnist data
	x=[]
	for i in xrange(1,501,1):	
		RF,ooberror=train_forest(train,i)
		ypred,ytest=test_forest(RF,test)
		correct1=error(ypred,ytest,0)
		x.append(i)
		correct.append(correct1)
		oobs.append(ooberror)
		print i
		print "WINE DATASET: \n" + " Correctness: " + str(correct1) + "%"


	print oobs
	plt.figure(2)
	plt.plot(x,oobs)
	plt.xlabel("# of Decision Trees in RF")
	plt.ylabel("Out of Bag Error")
	plt.show()


