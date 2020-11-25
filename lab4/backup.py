def buildDiscriminator():
    model = Sequential()
    model.add(Flatten(input_shape=IMAGE_SHAPE))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation= "sigmoid" ))
    inputTensor = Input(shape = IMAGE_SHAPE)
    return Model(inputTensor, model(inputTensor))

    def buildGenerator():
    model = Sequential()

    # TODO: build a generator which takes in a (NOISE_SIZE) noise array and outputs a fake
    #       mnist_f (28 x 28 x 1) image

    # Creating a Keras Model out of the network
    model.add(Dense(256 , input_dim = NOISE_SIZE))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(IMAGE_SIZE , activation = "tanh"))
    model.add(Reshape(IMAGE_SHAPE))
    inputTensor = Input(shape=(NOISE_SIZE,))
    return Model(inputTensor,model(inputTensor))


     # model = keras.Sequential()
    # inShape = (IH, IW, IZ)
    # lossType = keras.losses.sparse_categorical_crossentropy
    # opt = tf.train.AdamOptimizer()
   
    # model.add(keras.layers.Conv2D(32, kernel_size =(3, 3), activation = "relu", input_shape = inShape))
    # model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    # model.add(tf.keras.layers.Dropout(dropRate))


    # model.add(keras.layers.Conv2D(64, kernel_size =(3, 3), activation = "relu"))
    # model.add(keras.layers.MaxPooling2D(pool_size = (3,3)))

    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(128 , activation = "relu"))
    # model.add(keras.layers.Dense(NUM_CLASSES , activation = "softmax"))
    # model.compile(optimizer = opt, loss = lossType ,metrics=['accuracy'])

    # model.fit(x,y,epochs = 10)