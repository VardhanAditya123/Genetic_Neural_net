import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from sklearn import metrics


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
# tf.random.set_seed(1618)
tf.compat.v1.random.set_random_seed(1618)
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALGORITHM = "guesser"
ALGORITHM = "tf_net"
#ALGORITHM = "tf_conv"

# DATASET = "mnist_d"
DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    pass                                 # TODO: Add this case.
elif DATASET == "cifar_100_f":
    pass                                 # TODO: Add this case.
elif DATASET == "cifar_100_c":
    pass                                 # TODO: Add this case.


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
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


def buildTFNeuralNet(x, y, eps = 6):
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation = tf.nn.sigmoid),
    tf.keras.layers.Dense(10,activation = tf.nn.sigmoid),
    tf.keras.layers.Dense(10,activation = tf.nn.sigmoid),
    tf.keras.layers.Dense(10,activation = tf.nn.softmax)
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def trainANN(model,xTrain,yTrain,epochs=5):
    model.fit(xTrain,yTrain,epochs=10)
    return model

def runANN(data , model):
    (xTest, yTest) = data 
    answers = model.evaluate(xTest,yTest)
    print("loss:%f\naccuracy: %f" % tuple(answers))
    predictions = model.predict(xTest)
    return predictions

def printANN(data,predictions):
    (xTest, yTest) = data 
    yTestP = to_categorical(yTest, NUM_CLASSES)
    data = (xTest,yTestP)
    confMatrix(data,predictions)


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

def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
    pass        #TODO: Implement a CNN here. dropout option is required.
    return None

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        pass      # TODO: Add this case.
    elif DATASET == "cifar_100_f":
        pass      # TODO: Add this case.
    elif DATASET == "cifar_100_c":
        pass      # TODO: Add this case.
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain,xTest = xTrain/255.0 , xTest/255.0

    if ALGORITHM == "tf_net":
        return ((xTrain, yTrain), (xTest, yTest))

    elif ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
        
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
       
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
   
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
   
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = buildTFNeuralNet(xTrain, yTrain)
        model = trainANN(model,xTrain,yTrain,5)
        return model
   
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
   
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
   
   
    elif ALGORITHM == "tf_net":
        # print("Testing TF_NN.")
        # preds = model.predict(data)
        # for i in range(preds.shape[0]):
        #     oneHot = [0] * NUM_CLASSES
        #     oneHot[np.argmax(preds[i])] = 1
        #     preds[i] = oneHot
        # return preds
         return runANN(data,model)
   
   
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    if ALGORITHM == "tf_net":
        preds = runModel(data[1], model)
        # printANN(data[1],preds)
    else:
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)



if __name__ == '__main__':
    main()