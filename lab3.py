  
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import warnings

random.seed(1618)
np.random.seed(1618)
tf.compat.v1.random.set_random_seed(1618)
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "source.png"           #TODO: Add this.
STYLE_IMG_PATH = "style2.png"             #TODO: Add this.
OUTPUT_IMG_PATH = "output.png"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500
STYLE_IMG_H = 500
STYLE_IMG_W = 500
img_width = 500
img_height = 500
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
img_nrows = 500
img_ncols = 500
CHANNELS = 3

TRANSFER_ROUNDS = 10
numFilters = 20
STYLE_WEIGHT = 30
CONTENT_WEIGHT = 0.7
#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((CONTENT_IMG_H ,  CONTENT_IMG_W, 3))
    x = x[:, :, ::-1]
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def contentLoss(base, combination):
    return K.sum(K.square(combination - base))


def totalLoss(c_loss , s_loss):
    return (CONTENT_WEIGHT * c_loss)+(STYLE_WEIGHT * s_loss)

#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
   
    print("   Building transfer model.")
    

    # model = vgg19.VGG19(include_top =False, weights = "imagenet" , input_tensor = inputTensor)
    # model = vgg19.VGG19(include_top =False, weights = "imagenet")
    print("   Beginning transfer.")
    print("   VGG19 model loaded.")
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    
    genTensor = 0


    def compute_loss():
        
        inputTensor = K.concatenate([cData, sData, tData], axis=0)
        genTensor = tData
        model = vgg19.VGG19(include_top =False, weights = "imagenet" , input_tensor = inputTensor)
        outputDict = dict([(layer.name, layer.output) for layer in model.layers])
        loss = 0
        
        contentLayer = outputDict[contentLayerName]
        contentOutput = contentLayer[0, :, :, :]
        genOutput = contentLayer[2, :, :, :]
        c_loss = 0
        s_loss = 0

       
        c_loss = contentLoss(contentOutput , genOutput)
        for layerName in styleLayerNames:
            styleLayer = outputDict[layerName]
            styleOutput = styleLayer[1, :, :, :]
            genOutput = styleLayer[2, :, :, :]
            s_loss = styleLoss(styleOutput,genOutput) 

        loss = totalLoss(c_loss , s_loss)
        grads = K.gradients(loss, genTensor)
        return loss,grads
    
    
    combination_image = tf.Variable(tData)
    x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128
    x1 =  x.copy().reshape((1,img_height, img_width, 3))
    x1 = x1.astype("float64")
    x1 = tf.convert_to_tensor(x1)
    tData = x1
    opt = tf.train.AdamOptimizer()
   
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        # x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxiter=1)
        loss,grads = compute_loss()
        outputs = [loss]
        outputs += grads
        x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        outs = K.function([tData], outputs)([x])
        loss = outs[0]
        grads = numpy.array(grads)
        gradients = outs[1].reshape((1,img_height, img_width, 3))
        gradients=  gradients.astype("float64")
       
        # with tf.GradientTape() as tape:
        #     grads = tape.gradient(loss, combination_image)
        
        opt.apply_gradients([(gradients, tf.Variable(x1))])
       
       
        x1 =  x.copy().reshape((1,img_height, img_width, 3))
        x1 = x1.astype("float64")
        x1 = tf.convert_to_tensor(x1)
        tData = x1
        print('Current loss value:', loss)
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(x)
        img = array_to_img(img)
        saveFile = img.save( OUTPUT_IMG_PATH )   #TODO: Implement.
        # imsave(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
        print("   Transfer complete.")
       





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()