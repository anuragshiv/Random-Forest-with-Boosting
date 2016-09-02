import numpy as np
import sklearn as sk
from numpy import genfromtxt
import pandas as pd 
from sklearn import tree as dt
import random
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def read_dat(training_fp,testing_fp):
	#This method reads the dataset into pandas dataframes and inserts column names for ease of access.
	name=[]
	#MNIST

	train=pd.read_csv(training_fp)
	test=pd.read_csv(testing_fp)

	train=train.transpose()						# The MNIST data must be transposed before processing
	test=test.transpose()
	# Adding column names to MNIST Training and Test DataFrame 
	
	for i in range(train.shape[1]-1):
		name.append(str(i))
	name.append('label')
	train.columns= name
	test.columns=name

	return train,test

def testing(labels,alpha,Tm,M,test,correct1):
	print "TESTING"
	ytest=np.asarray(list(test.label))
	ytest=np.reshape(ytest,(test.shape[0],-1))
	test=test.drop('label',1)
	test_arr=np.asarray(test)
	count=0
	ypred=[]
	yactual=[]
	print ytest.shape
	for row,l in zip(test_arr,ytest):
		print l
		row=np.reshape(row,(1,test.shape[1]))
		sums=[]
		for k in labels:
			sum1=0
			for i in range(1,M):
				pred=Tm[i-1].predict(row)
				if(pred==k):
					sum1+=alpha[i-1]
			sums.append(sum1)
		sums=np.reshape(sums,(len(labels),-1))
		C= labels[np.argmax(sums)]
		ypred.append(C)
		yactual.append(l)
		if C==int(l):
			count+=1
	accuracy=float(count)/float(test_arr.shape[0])*100
	error.append(1-(correct/100))

	print accuracy
	print confusion_matrix(np.asarray(yactual), np.asarray(ypred))
	return error

def boosting(train,test,M,correct):

	y=np.asarray(list(train.label))
	y=np.reshape(y,(train.shape[0],-1))
	train1=train.drop('label',1)
	train_arr=np.asarray(train1)
	size=train1.shape[0]
	feats=train1.shape[1]
	n=float(1.0/(size))
	w=[n]*size
	alpha=[]
	Tm=[]
	ypred=[]
	print "BOOSTING"


	for m in range(1,M):
		tr=dt.DecisionTreeClassifier(max_depth=1,max_features=int(math.ceil(math.sqrt(feats))))
		tr=tr.fit(train_arr,y)
		err=0
		tot=0
		for i in range(0,size):
			row=train_arr[i]
			row=np.reshape(row,(1,train1.shape[1]))
			ypred.append(tr.predict(row))

		for i in range(0,size):			
			tot+=w[i]
			if(ypred[i]!=y[i]):
				err+= w[i]
		err=err/tot
		alph=(math.log((1-err)/err)) + math.log(feats-1)
		alpha.append(alph)

		for i in range(0,size):
			if(ypred[i]!=y[i]):
				w[i]=w[i]* math.exp(alph)
		w=np.reshape(w,(train.shape[0],1))

		Tm.append(tr)
	labels= np.unique(y)
	print "TRAINING COMPLETED"

	correct=testing(labels,alpha,Tm,M,test,correct)

	return correct
	

if __name__=="__main__":
	train,test=read_dat("train filepath","test filepath")
	correct=[]
	for M in x:
		error=boosting(train,test,M,correct)

	print error

	plt.figure(1)
	plt.plot(x,correct)
	plt.xlabel("# of Iterations (M)")
	plt.ylabel("Error")
	plt.show()
