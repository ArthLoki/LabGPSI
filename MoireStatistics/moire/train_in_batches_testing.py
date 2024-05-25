# 1 - Import Dependencies
import os
import tensorflow as tf
import numpy as np
import sys
import argparse
import cv2  # Importing OpenCV for image processing
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn import preprocessing
from skimage import io
from sklearn.model_selection import train_test_split
from mCNN import createModel
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


# Global constants
WIDTH = 500#384
HEIGHT = 375#512


# 2 - Definindo a função para ler e dimensionar imagens
def readAndScaleImage(f, customStr, trainImagePath):
    fileName = os.path.splitext(f)[0]
    fLL = os.path.join(trainImagePath, f.replace(fileName, fileName + customStr + '_LL').replace('.jpg', '.tiff'))
    fLH = os.path.join(trainImagePath, f.replace(fileName, fileName + customStr + '_LH').replace('.jpg', '.tiff'))
    fHL = os.path.join(trainImagePath, f.replace(fileName, fileName + customStr + '_HL').replace('.jpg', '.tiff'))
    fHH = os.path.join(trainImagePath, f.replace(fileName, fileName + customStr + '_HH').replace('.jpg', '.tiff'))

    try:
        imgLL = Image.open(fLL)
        imgLH = Image.open(fLH)
        imgHL = Image.open(fHL)
        imgHH = Image.open(fHH)
    except Exception as e:
        print('Error: Couldnt read the file {}. Make sure only images are present in the folder'.format(fileName))
        print('Exception:', e)
        return None

    imgLL = np.array(imgLL)
    imgLH = np.array(imgLH)
    imgHL = np.array(imgHL)
    imgHH = np.array(imgHH)
    imgLL = scaleData(imgLL, 0, 1)
    imgLH = scaleData(imgLH, -1, 1)
    imgHL = scaleData(imgHL, -1, 1)
    imgHH = scaleData(imgHH, -1, 1)

    return imgLL, imgLH, imgHL, imgHH


# 3 - Função para ler os dados de wavelet
def readWaveletData(imageFiles, trainImagePath, mode='train'):
    X_LL, X_LH, X_HL, X_HH, Y = [], [], [], [], []
    label = 0 if mode == 'train' else 1

    for f in imageFiles:
        imgLL, imgLH, imgHL, imgHH = readAndScaleImage(f, '', trainImagePath)
        if imgLL is not None:
            X_LL.append(imgLL.flatten())
            X_LH.append(imgLH.flatten())
            X_HL.append(imgHL.flatten())
            X_HH.append(imgHH.flatten())
            Y.append(label)

            # Ler imagens rotacionadas e invertidas
            imgLL, imgLH, imgHL, imgHH = readAndScaleImage(f, '_180', trainImagePath)
            X_LL.append(imgLL.flatten())
            X_LH.append(imgLH.flatten())
            X_HL.append(imgHL.flatten())
            X_HH.append(imgHH.flatten())
            Y.append(label)

            imgLL, imgLH, imgHL, imgHH = readAndScaleImage(f, '_180_FLIP', trainImagePath)
            X_LL.append(imgLL.flatten())
            X_LH.append(imgLH.flatten())
            X_HL.append(imgHL.flatten())
            X_HH.append(imgHH.flatten())
            Y.append(label)

    return np.array(X_LL), np.array(X_LH), np.array(X_HL), np.array(X_HH), np.array(Y)


# 4 - Função para escalar os dados
def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum, maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)
    return inp


def readAndScaleImage(f, customStr, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, sampleVal):
    fileName = (os.path.splitext(f)[0])
    fLL = (f.replace(fileName, fileName + customStr + '_LL')).replace('.jpg','.tiff')
    fLH = (f.replace(fileName, fileName + customStr + '_LH')).replace('.jpg','.tiff')
    fHL = (f.replace(fileName, fileName + customStr + '_HL')).replace('.jpg','.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg','.tiff')

    try:
        imgLL = Image.open(join(trainImagePath, fLL))
        imgLH = Image.open(join(trainImagePath, fLH))
        imgHL = Image.open(join(trainImagePath, fHL))
        imgHH = Image.open(join(trainImagePath, fHH))
    except Exception as e:
        print('Error: Couldnt read the file {}. Make sure only images are present in the folder'.format(fileName))
        print('Exception:', e)
        return None

    imgLL = np.array(imgLL)
    imgLH = np.array(imgLH)
    imgHL = np.array(imgHL)
    imgHH = np.array(imgHH)
    imgLL = scaleData(imgLL, 0, 1)
    imgLH = scaleData(imgLH, -1, 1)
    imgHL = scaleData(imgHL, -1, 1)
    imgHH = scaleData(imgHH, -1, 1)

    imgVector = imgLL.reshape(1, WIDTH*HEIGHT)
    X_LL[sampleIndex, :] = imgVector
    imgVector = imgLH.reshape(1, WIDTH*HEIGHT)
    X_LH[sampleIndex, :] = imgVector
    imgVector = imgHL.reshape(1, WIDTH*HEIGHT)
    X_HL[sampleIndex, :] = imgVector
    imgVector = imgHH.reshape(1, WIDTH*HEIGHT)
    X_HH[sampleIndex, :] = imgVector

    Y[sampleIndex, 0] = sampleVal;
    X_index[sampleIndex, 0] = sampleIndex;

    return True


# 5 - Função para dividir os dados de treinamento e teste
def splitTrainTestData(X_LL, X_LH, X_HL, X_HH, Y, test_size=0.1):
    return train_test_split(X_LL, X_LH, X_HL, X_HH, Y, test_size=test_size, random_state=42, stratify=Y)


# 6 - Função para criar os geradores de dados
def createDataGenerators(X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train, X_LL_test, X_LH_test, X_HL_test, X_HH_test, Y_test, batch_size=32):
    train_steps = len(Y_train) // batch_size
    val_steps = len(Y_test) // batch_size

    def train_generator():
        while True:
            for i in range(train_steps):
                yield [X_LL_train[i * batch_size:(i + 1) * batch_size],
                       X_LH_train[i * batch_size:(i + 1) * batch_size],
                       X_HL_train[i * batch_size:(i + 1) * batch_size],
                       X_HH_train[i * batch_size:(i + 1) * batch_size]], \
                      Y_train[i * batch_size:(i + 1) * batch_size]

    def val_generator():
        while True:
            for i in range(val_steps):
                yield [X_LL_test[i * batch_size:(i + 1) * batch_size],
                       X_LH_test[i * batch_size:(i + 1) * batch_size],
                       X_HL_test[i * batch_size:(i + 1) * batch_size],
                       X_HH_test[i * batch_size:(i + 1) * batch_size]], \
                      Y_test[i * batch_size:(i + 1) * batch_size]

    return train_generator(), val_generator(), train_steps, val_steps


# 7 - Função para treinar o modelo CNN
def trainCNNModel(train_gen, train_steps, val_gen, val_steps, test_gen, height, width, depth, num_classes, num_epochs):
    strategy = tf.distribute.MirroredStrategy()  # Definindo a estratégia de múltiplas GPUs
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    batch_size = 32 * strategy.num_replicas_in_sync  # Ajustar o tamanho do batch de acordo com o número de GPUs

    with strategy.scope():  # Aplicação da estratégia de distribuição
        model = createModel(height, width, depth, num_classes)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkPointFolder = 'checkPoint'
    if not os.path.exists(checkPointFolder):
        os.makedirs(checkPointFolder)
    checkpoint_name = checkPointFolder + '/Weights-{epoch:03d}--{val_loss:.5f}.keras'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    # Treinamento do modelo usando geradores
    model.fit(train_gen,
              steps_per_epoch=train_steps,
              epochs=num_epochs, verbose=1,
              validation_data=val_gen,
              validation_steps=val_steps,
              callbacks=callbacks_list)

    # Avaliação do modelo usando o gerador de teste
    score, acc = model.evaluate(test_gen, steps=val_steps, verbose=1)

    # Salvar o modelo
    model.save('moirePattern3CNN_.keras')

    return model


# 8 - Main
def main(args):
    positiveImagePath = args.positiveImages
    negativeImagePath = args.negativeImages
    numEpochs = args.epochs
    positiveTrainImagePath = args.trainingDataPositive
    negativeTrainImagePath = args.trainingDataNegative
    return


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())
    main(parse_arguments(sys.argv[1:]))

    # Chamada à função com os caminhos corretos
    data_gen, total_samples = readWaveletData(positiveImagePath, negativeImagePath)