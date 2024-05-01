import os
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Add, Multiply, Maximum

def createModel(height, width, depth):
    kernel_size_1 = 7  # Tamanho do kernel para a primeira camada convolucional
    kernel_size_2 = 3  # Tamanho do kernel para camadas convolucionais subsequentes
    pool_size = 2  # Tamanho do pooling para as primeiras camadas
    conv_depth_1 = 32  # Profundidade para a primeira camada convolucional
    conv_depth_2 = 64  # Profundidade para a segunda camada convolucional
    conv_depth_3 = 128  # Profundidade para a terceira camada convolucional
    drop_prob_1 = 0.25  # Probabilidade de dropout após o pooling
    drop_prob_2 = 0.5  # Probabilidade de dropout na camada totalmente conectada
    hidden_size = 32  # Tamanho da camada totalmente conectada

    # Entradas para diferentes bandas da transformada de Haar
    inpLL = Input(shape=(height, width, depth))
    inpLH = Input(shape=(height, width, depth))
    inpHL = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))

    # Primeira camada convolucional aplicada a todas as entradas
    conv_1 = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')
    conv_1_LL = conv_1(inpLL)
    conv_1_LH = conv_1(inpLH)
    conv_1_HL = conv_1(inpHL)
    conv_1_HH = conv_1(inpHH)

    # Pooling aplicado a todas as saídas da primeira camada convolucional
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))
    pool_1_LL = pool_1(conv_1_LL)
    pool_1_LH = pool_1(conv_1_LH)
    pool_1_HL = pool_1(conv_1_HL)
    pool_1_HH = pool_1(conv_1_HH)

    # Combinação das saídas do pooling e segunda camada convolucional
    combined = Maximum()([pool_1_LL, pool_1_LH, pool_1_HL, pool_1_HH])
    conv_2 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(combined)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

    # Terceira camada convolucional
    conv_3 = Convolution2D(conv_depth_3, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)

    # Camada totalmente conectada e saída
    flat = Flatten()(pool_3)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(1, activation='sigmoid')(drop_3)  # Camada de saída para classificação binária

    model = Model(inputs=[inpLL, inpLH, inpHL, inpHH], outputs=out)  # Definindo o modelo com suas entradas e saídas

    return model

# Lembre-se de compilar o modelo depois de criá-lo
# model = createModel(height, width, depth)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
