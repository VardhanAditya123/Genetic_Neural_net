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
# For N layer custom net
NO_OF_LAYERS = 3
NEURONS_PER_LAYER = 512
no_of_generations = 3
no_of_individuals = 20
mutate_factor = 0.1
NETCOUNT = 1
ALGORITHM = "custom_net"
elites = 5
losers = 3

if ALGORITHM == "custom_net":
    print("\nNumber of layers: %d" % NO_OF_LAYERS)
    print("Neurons per Layer: %d" % NEURONS_PER_LAYER)
    print("Type of algorithm: " + ALGORITHM)
    print()



#=========================<NETWORK FUNCTIONS>================================================

class NeuralNetwork_NLayer():
    def __init__(self,custom, inputSize, outputSize, neuronsPerLayer, layers, netcount , learningRate = 0.1):
        self.accuracy = 0
        self.name = "n"+ str(netcount)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W = []
        self.L = []
        self.Z = []
        self.delta=[]
        self.N_layers = layers
        self.lc = 0
        self.loss = 0
        # self.tloss = 0
        i = 0
        if(custom == 0):
            while i < layers:
                if i == 0:
                    self.W.append(np.random.randn(self.neuronsPerLayer, self.inputSize))
                elif i == (layers - 1):
                    self.W.append(np.random.randn(self.outputSize ,self.neuronsPerLayer))
                else:
                    self.W.append(np.random.randn(self.neuronsPerLayer ,self.neuronsPerLayer))
                i+=1
        
    



    def addLayer(self,layer,parentA,parentB,index):

        gene = layer.copy()
        rows = gene.shape[0]
        cols = gene.shape[1]
        A = parentA.W[index]
        B = parentB.W[index]

        if(index != 0 and index!= NO_OF_LAYERS -1 ):
            n = np.random.randint(1,NO_OF_LAYERS-1)
            A = parentA.W[n]
            n = np.random.randint(1,NO_OF_LAYERS-1)
            B = parentB.W[n]

        for i in range(0, rows):
            for j in range(0, cols):
                n = np.random.rand()
                if(n <0.5):
                    gene[i][j] =A[i][j]
                else:
                    gene[i][j] =B[i][j]

        self.W.append(gene)
        self.lc += 1


    # Activation function.
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def __relu(self, x):
        return max(0,x.all())

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sig_x = self.__sigmoid(x) 
        sig_d = sig_x * (1 - sig_x)
        return sig_d

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]




    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        pass


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
    def predict_N(self, xVals):
        L = self.__forward(xVals)
        return L[self.N_layers - 1]

   


    
#=========================<GENE Functions>==================================


     #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    
def mutate(new_individual):
    # print("MUTATING FOR THIS GEN")
    # print("MUTATING" + str(new_individual.name))
    for x in range (NO_OF_LAYERS):
        gene = new_individual.W[x]
        rows = gene.shape[0]
        cols = gene.shape[1]
        for i in range(0, rows):
            for j in range(0, cols):
                n = np.random.rand()
                if(n < mutate_factor):
                    (new_individual.W[x])[i,j] =  np.random.rand()
    return new_individual



def crossover(individuals):
    global NETCOUNT
    new_individuals = []
    for x in range (elites):
        new_individuals.append(individuals[x])
    
    for y in range(elites, elites + losers):
        new_individuals.append(mutate(individuals[y]))

    start = elites + losers
    for i in range(start, no_of_individuals):
        a = np.random.randint(elites)
        parentA = individuals[a]
        while(1):
            b = np.random.randint(start)
            if(b != a):
                parentB = individuals[b]
                break

        CUSTOM = 1
        new_individual = NeuralNetwork_NLayer(CUSTOM,IMAGE_SIZE,NUM_CLASSES,NEURONS_PER_LAYER,NO_OF_LAYERS,NETCOUNT,0.1)
        NETCOUNT+=1

        for j in range(NO_OF_LAYERS):
            Br = parentB.W[j]
            new_individual.addLayer(Br,parentA,parentB,j)

        new_individual = mutate(new_individual)
        new_individuals.append(new_individual)
    return new_individuals

def train_nets(data,individuals):
    i = 0
    (xTrain , yTrain) = data[0]
    while i < 50:
        for individual in individuals:
            x = xTrain[i]
            y = yTrain[i]
            L  = individual.predict_N(x)
            ind = findMax(y)
            individual.loss += y[ind] - L[ind]
        i+=1
        indviduals = evolve(individuals)
    return individuals


#=========================<HELPER Functions>==================================


def Customclassifier(xTest , model):
    ans = []
    for x in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        check = (model.predict_N(x)).flatten()
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
    
 
def confMatrix(data, preds):
    xTest, yTest = data
    n_preds=[]
    n_yTest=[]
    for i in range(preds.shape[0]):
        n_preds.append(findMax(preds[i] ))
        n_yTest.append(findMax(yTest[i] ))
    labels = [0,1,2,3,4,5,6,7,8,9]
    confusion = metrics.confusion_matrix(n_yTest, n_preds,labels)
    report = metrics.classification_report(n_yTest, n_preds,labels)
    print("\nConfusion Matrix:\n")
    print(confusion)
    print("\nReport:")
    print(report)   

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
    xTrain = xTrain.reshape(60000,784,1)
    xTest = xTest.reshape(10000,784,1)
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    # yTrainP = yTrainP.reshape(60000,10,1)
    print("\nAfter preprocessing:")
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP)) 



def buildModel():
     CUSTOM = 0
     global NETCOUNT
     n1 = NeuralNetwork_NLayer(CUSTOM,IMAGE_SIZE,NUM_CLASSES,NEURONS_PER_LAYER,NO_OF_LAYERS,NETCOUNT,0.1) 
     NETCOUNT+=1
     return n1

def trainModel(data , model):
    xTrain, yTrain = data
    model.train(xTrain,yTrain)
    return model


def runModel(data, model):
    return Customclassifier(data , model)




def evalResults(data, preds , individual):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    individual.accuracy = accuracy
    # confMatrix(data,preds)
    print("NAME: " + str(individual.name))
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()

    
def trainModels(data , individuals):
    for i  in range (len(individuals)):
        individuals[i] = trainModel(data[0] , individuals[i])
    return individuals

def runModels (data , individuals):
    for individual in individuals: 
        preds = runModel(data[0][0],individual)
        evalResults(data[0], preds , individual)
    return individuals

def evolve(individuals):
    individuals = sorted(individuals, key=lambda x: x.accuracy, reverse=True)
    new_individuals = crossover(individuals)
    return new_individuals
#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)

    individuals = []
    for i in range(no_of_individuals):
        model = buildModel()
        individuals.append(model)
    
    for generation in range(no_of_generations):
        print("================<NEXT GENERATION>===================")
        individuals = runModels(data, individuals)
        individuals = evolve(individuals)

    individuals = sorted(individuals, key=lambda x: x.accuracy, reverse=True)
    model = individuals[0]
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds ,model)
    confMatrix(data[1],preds)
    


if __name__ == '__main__':
    main()

# END_OF_LAB5