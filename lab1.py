#!/usr/bin/python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from sklearn import metrics




# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.compat.v1.set_random_seed(1618)
tf.compat.v1.random.set_random_seed(1618)
# tf.random.set_random_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_NLayer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, layers ,learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W = []
        self.L = []
        self.Z = []
        self.delta=[]
        self.N_layers = layers
        i = 0
        while i < layers:
            if i == 0:
                self.W.append(np.random.randn(self.neuronsPerLayer, self.inputSize))
            elif i == (layers - 1):
                self.W.append(np.random.randn(self.outputSize ,self.neuronsPerLayer))
            else:
                self.W.append(np.random.randn(self.neuronsPerLayer ,self.neuronsPerLayer))
            i+=1
            
            
    

    # Activation function.
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sig_x = self.__sigmoid(x) 
        sig_d = sig_x * (1 - sig_x)
        return sig_d

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        i = 0
        layer = self.N_layers
        print("Number of layers: %d" % layer)
        while i < 60000:
            x = xVals[i]
            y = yVals[i]
            L  = self.__forward(x)
            Z = self.Z
            j = layer -1
            self.delta={}
            while j >=0:
                if(j == layer -1):
                    self.delta.update( {j : (L[j] - y)*self.__sigmoidDerivative(Z[j])} )
                else:
                    self.delta.update({j :np.dot(  self.W[j+1].T ,  (self.delta[j+1]) ) * self.__sigmoidDerivative(Z[j])})
                j-=1
            
            j = 0
           
            while j < layer:
                 if j == 0:
                     self.W[j] -= (self.delta[j]).dot(x.T)   
                 else:
                     self.W[j] -= (self.delta[j]).dot(L[j-1].T)
                 j+=1
            i+=1


     #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input ):
        
        self.L=[]
        self.Z=[]
        i = 0
        while i < self.N_layers:
            if i == 0:
                Z = np.dot(self.W[i] , input)
                self.Z.append(Z)
                self.L.append(self.__sigmoid(Z))
            else:
                Z = np.dot(self.W[i] , self.L[i-1])
                self.Z.append(Z)
                self.L.append(self.__sigmoid(Z))
            i+=1
        # return layer1, layer2
        return self.L


    # Predict.
    def predict(self, xVals):
        L = self.__forward(xVals)
        return L[self.N_layers - 1]




# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

def Customclassifier(xTest , model):
    ans = []
    for x in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        check = (model.predict(x)).flatten()
        index = findMax(check)
        pred[index] = 1
        ans.append(pred)
        
    return np.array(ans)

def findMax(layer):
    max = layer[0]
    max_l = 0
    i = 0
    while i < 10:
        if layer[i] > max:
            max = layer[i]
            max_l = i
        i+=1
    return max_l


def buildANN():
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(60,activation = tf.nn.sigmoid),
    tf.keras.layers.Dense(10,activation = tf.nn.sigmoid)])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def trainANN(model,xTrain,yTrain,epochs=5):
    model.fit(xTrain,yTrain,epochs=5)
    return model

def runANN(data , model):
    (xTest, yTest) = data 
    preds = model.evaluate(xTest,yTest)
    print("loss:%f\naccuracy: %f" % tuple(preds))
    

#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))




def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain,xTest = xTrain/255.0 , xTest/255.0
    
    if ALGORITHM == "custom_net":
        xTrain = xTrain.reshape(60000,784,1)
        xTest = xTest.reshape(10000,784,1)
        yTrainP = to_categorical(yTrain, NUM_CLASSES)
        yTestP = to_categorical(yTest, NUM_CLASSES)
        yTrainP = yTrainP.reshape(60000,10,1)
        print("New shape of xTrain dataset: %s." % str(xTrain.shape))
        print("New shape of xTest dataset: %s." % str(xTest.shape))
        print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
        print("New shape of yTest dataset: %s." % str(yTestP.shape))
        return ((xTrain, yTrainP), (xTest, yTestP)) 

    else:
        return ((xTrain, yTrain), (xTest, yTest))
    


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
   
    elif ALGORITHM == "custom_net":
        n1 = NeuralNetwork_NLayer(784,10,50,2,0.1) 
        n1.train(xTrain,yTrain)
        return n1

    elif ALGORITHM == "tf_net":
        model = buildANN()
        model = trainANN(model,xTrain,yTrain,5)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        return Customclassifier(data , model)
    elif ALGORITHM == "tf_net":
        return runANN(data,model)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    confMatrix(data,preds)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()

def confMatrix(data, preds):
    xTest, yTest = data
    n_preds=[]
    n_yTest=[]
    for i in range(preds.shape[0]):
        n_preds.append(findMax(preds[i] ))
        n_yTest.append(findMax(yTest[i] ))
    # labels = ['0','1','2','3','4','5','6','7','8','9'] 
    labels = [0,1,2,3,4,5,6,7,8,9]
    confusion = metrics.confusion_matrix(n_yTest, n_preds,labels)
    report = metrics.classification_report(n_yTest, n_preds,labels)
    print("\nConfusion Matrix:\n")
    print(confusion)
    print("\nReport:")
    print(report)
    


#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    if ALGORITHM == "tf_net":
        runModel(data[1], model)
    else:
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)


if __name__ == '__main__':
    main()