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

mnist = tf.keras.datasets.mnist
no_of_generations = 15
no_of_individuals = 10
mutate_factor = 0.1
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

input_shape =(28, 28,1)

def init():
    model = keras.Sequential()
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(60,activation = tf.nn.sigmoid),
    tf.keras.layers.Dense(10,activation = tf.nn.sigmoid)])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train(models):
  
    losses = []
     
    for i in range(len(models)):
        history = models[i].fit(x=X_train,y=y_train, epochs=1, validation_data=(X_test, y_test))
        losses.append(round(history.history['loss'][-1], 4))
        
    return models, losses

# Depending on the Application the number of generations may or maynot be fixed

def mutate(new_individual):
  genes = []
  for gene in new_individual.layers:
    n = random.random()
    if(n < mutate_factor):
      #Assign random values to certain genes within the maximum acceptable bounds
      genes.append(random.random())
    else:
      genes.append(gene)
      
  return genes


def crossover(individuals):
    new_individuals = []
    print("INDIVID:")
    print(type(individuals))
    print(len(individuals))
    for i in range(2, no_of_individuals):
        new_individual = []

        if(i < (no_of_individuals - 2)):
            if(i == 2):
                parentA = random.choice(individuals[:3])
                parentB = random.choice(individuals[:3])
            else:
                parentA = random.choice(individuals[:])
                parentB = random.choice(individuals[:])

            # for layer in parentA.layers: print(layer.get_config(), layer.get_weights())
            # for i in range(len(parentA)):
            for layerA ,layerB in zip(parentA.layers , parentB.layers):
                n = random.random()
                if(n< 0.5):
                    new_individual.append(layerA)
                else:
                    new_individual.append(layerB)
         
        else:
            new_individual = random.choice(individuals[:])

        # new_individuals.append(mutate(new_individual))
        new_individuals.append(new_individual)

    newi = keras.Sequential()
    for layer in new_individuals:
        newi.add(tf.keras.layers(layer))
    new_individuals = newi
    return new_individuals


def evolve(individuals, fitness):
    sorted_y_idx_list = sorted(range(len(fitness)),key=lambda x:fitness[x])
    individuals = [individuals[i] for i in sorted_y_idx_list ]
    individuals.reverse()
    new_individuals = crossover(individuals)
    return new_individuals

# to initialize the beginning generation
def main():
    individuals = []
    for i in range(2):
        individuals.append(init())

    # control loop
    for generation in range(2):
        individuals, losses = train(individuals)
        print(losses)
        individuals = evolve(individuals, losses)
    
if __name__ == '__main__':
    main()