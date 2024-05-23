import os
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Add, Multiply, Maximum, BatchNormalization
from keras.initializers import HeNormal

def conv_block(inp, depth, kernel_size, pool_size, initializer):
    conv = Convolution2D(depth, (kernel_size, kernel_size), padding='same', activation='relu', kernel_initializer=initializer)(inp)
    batch_norm = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm)
    return pool

def createModel(height, width, depth, num_classes):
    kernel_size_1 = 7
    kernel_size_2 = 3
    pool_size = 2
    conv_depth_1 = 32
    conv_depth_2 = 16
    drop_prob_1 = 0.25
    drop_prob_2 = 0.5
    hidden_size = 32
    initializer = HeNormal()

    inpLL = Input(shape=(height, width, depth))
    inpLH = Input(shape=(height, width, depth))
    inpHL = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))

    # Aplicando a função conv_block para cada entrada
    pool_1_LL = conv_block(inpLL, conv_depth_1, kernel_size_1, pool_size, initializer)
    pool_1_LH = conv_block(inpLH, conv_depth_1, kernel_size_1, pool_size, initializer)
    pool_1_HL = conv_block(inpHL, conv_depth_1, kernel_size_1, pool_size, initializer)
    pool_1_HH = conv_block(inpHH, conv_depth_1, kernel_size_1, pool_size, initializer)

    # Merging and further layers
    avg_LH_HL_HH = Maximum()([pool_1_LH, pool_1_HL, pool_1_HH])
    inp_merged = Multiply()([pool_1_LL, avg_LH_HL_HH])

    C4 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(inp_merged)
    batch_norm_C4 = BatchNormalization()(C4)
    S2 = MaxPooling2D(pool_size=(4, 4))(batch_norm_C4)
    drop_1 = Dropout(drop_prob_1)(S2)

    C5 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(drop_1)
    batch_norm_C5 = BatchNormalization()(C5)
    S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_C5)

    C6 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(S3)
    batch_norm_C6 = BatchNormalization()(C6)
    S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_C6)
    drop_2 = Dropout(drop_prob_1)(S4)

    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(1, activation='sigmoid')(drop_3)

    model = Model(inputs=[inpLL, inpLH, inpHL, inpHH], outputs=out)

    return model
