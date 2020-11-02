import numpy as np
import csv
import scipy.io as spio

class MNIST:
	def __init__(self):
		data = spio.loadmat('mnistReduced.mat')

		self.xTrain = data['images_train']  # 784*30000
		self.yTrain = data['labels_train']  # 1*30000

		self.xVal = data['images_val']  # 784*3000
		self.yVal = data['labels_val'] # 1*3000

		self.xTest = data['images_test'] # 784*3000
		self.yTest = data['labels_test'] # 1*3000
		
def normalize(array):
	array = 2.*array/255.-1.
	return array

def hotEncode(array, N):
	array = np.eye(N)[array,:]
	return array[0].T

def overFit(arr1, arr2):
	for i in range(0, len(arr1)-1):
		if arr1[i+1] - arr1[i] > 0:
			return False
		if arr2[i+1] - arr2[i] < 0:
			return False

	return True

def forwardPass(trainInput, trainLabels, W1, b1, W2, b2):
	z1 = np.dot(W1, trainInput)+b1
	h1 = np.maximum(z1, 0)
	z2 = np.dot(W2, h1)+b2

	c = np.amax(z2)
	yhat = np.exp(z2-c)/np.sum(np.exp(z2-c), axis=0)

	N = np.size(trainInput, 1)
	trainingLoss = -1*np.sum(trainLabels*np.log(yhat))/N
	accuracy = (np.argmax(trainLabels,axis=0) == np.argmax(yhat,axis=0)).mean()
	return z1, z2, h1, yhat, trainingLoss, accuracy

def backwardPass(trainInput, trainLabels, W1, W2, b1, b2, yhat, h1, z1, \
		learningRate):

	L2 = yhat - trainLabels
	gradW2 = np.dot(L2, h1.transpose())/L2.shape[1]
	gradb2 = np.mean(L2, axis=1)[np.newaxis].transpose()
	dz1 = h1
	dz1[dz1 > 0] = 1
	L1 = (np.dot(W2.transpose(), L2))*dz1
	gradW1 = np.dot(L1, trainInput.transpose())/L1.shape[1]
	gradb1 = np.mean(L1, axis=1)[np.newaxis].transpose()

	newW1 = W1 - learningRate*gradW1
	newW2 = W2 - learningRate*gradW2

	newb1 = b1 - learningRate*gradb1
	newb2 = b2 - learningRate*gradb2

	return gradW1, gradb1, gradW2, gradb2, newW1, newb1, newW2, newb2

if __name__ == "__main__":
	mnist = MNIST()
	# x is values from image
	# y is class label
	# xTrain, yTrain
	# xVal, yVal
	# xTest, yTest

	# table of frequency of class labels in training data
	unique, counts = np.unique(mnist.yTrain, return_counts=True)
	table = dict(zip(unique, counts))
	print(table)
	print(mnist.yTrain)

	# normalize data [-1, 1]
	mnist.xTrain = normalize(mnist.xTrain)	
	mnist.xVal = normalize(mnist.xVal)
	mnist.xTest = normalize(mnist.xTest)
	
	# hot encode class labels
	mnist.yTrain = hotEncode(mnist.yTrain, 10)
	mnist.yVal = hotEncode(mnist.yVal, 10)
	mnist.yTest = hotEncode(mnist.yTest, 10)
	print(mnist.xTrain.shape)
	print(mnist.yTrain.shape)
	batches = []
	for i in range(0, 118):
		batches.append((mnist.xTrain[:,i*256:(i+1)*256], \
				mnist.yTrain[:,i*256:(i+1)*256]))
	nInput = 784
	nHidden = 30
	nOutput = 10
	
	# Optimization of Learning Rate
	trainingLossTrainArr = []
	accuracyTrainArr = []
	trainingLossValArr = []
	accuracyValArr = []

	learningRates = [0.001, 0.01, 0.1, 1, 10]
	for idx, rate in enumerate(learningRates):
		W1 = 0.001*np.random.randn(nHidden, nInput)
		b1 = np.zeros((nHidden, 1))
		W2 = 0.001*np.random.randn(nOutput, nHidden)
		b2 = np.zeros((nOutput, 1))
		for i in range(0, 100):
			for batch in batches:
				z1, z2, h1, yhat, trainingLoss, accuracy = \
						forwardPass(batch[0], batch[1], W1, b1, W2, b2)
				gradW1, gradb1, gradW2, gradb2, W1, b1, W2, b2 = \
						backwardPass(batch[0], batch[1], \
						W1, W2, b1, b2, yhat, h1, z1, rate)

			z1, z2, h1, yhat, trainingLossTrain, accuracyTrain = \
					forwardPass(mnist.xTrain, mnist.yTrain, W1, b1, W2, b2)
			z1, z2, h1, yhat, trainingLossVal, accuracyVal = \
					forwardPass(mnist.xVal, mnist.yVal, W1, b1, W2, b2)

			trainingLossTrainArr.append(trainingLossTrain)
			accuracyTrainArr.append(accuracyTrain)
			trainingLossValArr.append(trainingLossVal)
			accuracyValArr.append(accuracyVal)
			print(i)
		# writing to csv file	
		resultFile = open("rate"+str(idx)+".csv", 'w')
		resultFile.write("TrainLoss, TrainAcc, ValLoss, ValAcc \n")
		for i in range(0, len(trainingLossTrainArr)):
			resultFile.write(str(trainingLossTrainArr[i])+",")
			resultFile.write(str(accuracyTrainArr[i])+",")
			resultFile.write(str(trainingLossValArr[i])+",")
			resultFile.write(str(accuracyValArr[i])+",")
			resultFile.write("\n")
		resultFile.close()

	# Early Stopping
	TTLArr = []
	TAArr = []
	VTLArr = []
	VAArr = []

	W1 = 0.001*np.random.randn(nHidden, nInput)
	b1 = np.zeros((nHidden, 1))
	W2 = 0.001*np.random.randn(nOutput, nHidden)
	b2 = np.zeros((nOutput, 1))
	learningRate = 0.1
	count = 0
	while True:
		count += 1
		for batch in batches:
			z1, z2, h1, yhat, trainingLoss, accuracy = \
					forwardPass(batch[0], batch[1], W1, b1, W2, b2)
			gradW1, gradb1, gradW2, gradb2, W1, b1, W2, b2 = \
					backwardPass(batch[0], batch[1], \
					W1, W2, b1, b2, yhat, h1, z1, learningRate)

		z1, z2, h1, yhat, trainingLossTrain, accuracyTrain = \
				forwardPass(mnist.xTrain, mnist.yTrain, W1, b1, W2, b2)
		z1, z2, h1, yhat, trainingLossVal, accuracyVal = \
				forwardPass(mnist.xVal, mnist.yVal, W1, b1, W2, b2)
			
		TTLArr.append(trainingLossTrain)
		TAArr.append(accuracyTrain)
		VTLArr.append(trainingLossVal)
		VAArr.append(accuracyVal)
	
		if len(TTLArr) >= 5:
			if overFit(TTLArr[-5:], VTLArr[-5:]):
				break	

	resultFile = open("earlystopping.csv", 'w')
	resultFile.write("TrainLoss, TrainAcc, ValLoss, ValAcc \n")
	for i in range(0, len(TTLArr)):
		resultFile.write(str(TTLArr[i])+",")
		resultFile.write(str(TAArr[i])+",")
		resultFile.write(str(VTLArr[i])+",")
		resultFile.write(str(VAArr[i])+",")
		resultFile.write("\n")
	resultFile.close()

	# Testing Loss and Testing Accuracy
	z1, z2, h1, yhat, trainingLossTest, accuracyTest = \
			forwardPass(mnist.xTest, mnist.yTest, W1, b1, W2, b2)

	print("Test:")
	print("Loss = "+str(trainingLossTest))
	print("Acc  = "+str(accuracyTest))
