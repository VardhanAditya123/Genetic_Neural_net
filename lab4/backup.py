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