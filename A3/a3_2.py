import tensorflow.keras as keras
import tensorflow as tf
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib as pyplot
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD



def load_dataset():
	#load dataset
	(trainX,trainY),(testX,testY)=mnist.load_data()

	#reshape
	trainX=trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX=testX.reshape((testX.shape[0], 28, 28, 1))

	trainY=to_categorical(trainY)
	testY=to_categorical(testY)
	return trainX,trainY,testX,testY
	
	
#scale-normalization	
def prp_pixels(train,test):
	train_normal=train.astype('float32')
	test_normal=test.astype('float32')	
	train_normal=train_normal/255.0
	test_normal=test_normal/255.0
	return train_normal,test_normal
	
	
	
#define model
def nn_model():
	model=tf.keras.models.Sequential()

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(576,input_dim=5760,activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(10,activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(20,activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
	opt=SGD(lr=0.001,momentum=0.6)
	model.compile(optimizer='SGD', loss='categorical_crossentropy', 	metrics=['accuracy'])
	return model
	
	
	
	
	
	
	
#k-fold evaluation	
def evaluate_model(dataX,dataY,n_folds=5):
	scores,histories,ce_loss=list(),list(),list()
	kfold=KFold(n_folds,shuffle=True, random_state=1)
	for train_ix,test_ix in kfold.split(dataX):
		model=nn_model()
		trainX,trainY,testX,testY=dataX[train_ix],dataY[train_ix],dataX[test_ix],dataY[test_ix]
		history=model.fit(trainX,trainY,epochs=10,batch_size=128,validation_data=(testX,testY),verbose=0)
		_,acc=model.evaluate(testX,testY,verbose=0)
		print('> %.3f' %(acc*100.0)  )


		scores.append(acc)
		histories.append(history)

		#print('---- loss= %.5f' % histories[n_folds].history['loss'])
	
	return scores,histories,ce_loss
	
	
def diagnostics(histories):
	for i in range(len(histories)):
	#plot ce
		plt.subplot(2,1,1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'],color='blue',label='train')
		plt.plot(histories[i].history['val_loss'],color='red',label='test')
		#plot accuracy
		plt.subplot(2,1,2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'],color='blue',label='train')
		plt.plot(histories[i].history['val_accuracy'],color='red',label='test')		
	plt.show()
		
def perfomance(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' %(mean(scores)*100,std(scores)*100,len(scores)))

	plt.boxplot(scores)
	plt.show()
def loss(histories):
	for i in range(len(histories)):
		
		print('CE loss= %.5f ' %(mean(histories[i].history['loss'])))	
	
def run_test():
	trainX,trainY,testX,testY=load_dataset()
	trainX,testX=prp_pixels(trainX,trainY)
	scores,histories,ce_loss=evaluate_model(trainX,trainY)
	diagnostics(histories)
	perfomance(scores)
	loss(histories)
	
run_test()

	
	
	
