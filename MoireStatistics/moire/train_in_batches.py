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
from DataGenerator import DataGenerator

# Constants
WIDTH = 500#384
HEIGHT = 375#512


def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum, maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)

    return inp


# As funções readWaveletData, trainCNNModel, e evaluate devem ser definidas para lidar com geradores
# e configuração apropriada de treinamento, validação e teste
def readAndScaleImage(f, customStr, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, sampleVal):
    fileName = (os.path.splitext(f)[0])
    fLL = (f.replace(fileName, fileName + customStr + '_LL')).replace('.jpg', '.tiff')
    fLH = (f.replace(fileName, fileName + customStr + '_LH')).replace('.jpg', '.tiff')
    fHL = (f.replace(fileName, fileName + customStr + '_HL')).replace('.jpg', '.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg', '.tiff')

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

    imgVector = imgLL.reshape(1, WIDTH * HEIGHT)
    X_LL[sampleIndex, :] = imgVector
    imgVector = imgLH.reshape(1, WIDTH * HEIGHT)
    X_LH[sampleIndex, :] = imgVector
    imgVector = imgHL.reshape(1, WIDTH * HEIGHT)
    X_HL[sampleIndex, :] = imgVector
    imgVector = imgHH.reshape(1, WIDTH * HEIGHT)
    X_HH[sampleIndex, :] = imgVector

    Y[sampleIndex, 0] = sampleVal;
    X_index[sampleIndex, 0] = sampleIndex;

    return True


def readImageSet(imageFiles, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, bClass):

    for f in imageFiles:
        ret = readAndScaleImage(f, '', trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, bClass)
        if ret == True:
            sampleIndex = sampleIndex + 1

        #read 180deg rotated data
        ret = readAndScaleImage(f, '_180', trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex,bClass)
        if ret == True:
            sampleIndex = sampleIndex + 1

        #read 180deg FLIP data
        ret = readAndScaleImage(f, '_180_FLIP', trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, bClass)
        if ret == True:
            sampleIndex = sampleIndex + 1

    return sampleIndex





def readWaveletData(positiveImagePath, negativeImagePath, mode='train', batch_size=32):
    # Usage example:
    # train_gen = readWaveletData('path/to/positive/images', 'path/to/negative/images', mode='train')
    # val_gen = readWaveletData('path/to/positive/images', 'path/to/negative/images', mode='validation')
    # test_gen = readWaveletData('path/to/positive/images', 'path/to/negative/images', mode='test')

    # Determine the correct path based on the mode
    if mode == 'train':
        imagePath = positiveImagePath
        label = 0
    else:
        imagePath = negativeImagePath
        label = 1

    # List all image files
    positiveImageFiles = [f for f in listdir(positiveImagePath) if isfile(join(positiveImagePath, f))]
    negativeImageFiles = [f for f in listdir(negativeImagePath) if isfile(join(negativeImagePath, f))]

    # Shuffle the list of files to ensure random order
    np.random.shuffle(imageFiles)

    # Generator function
    def data_generator():
        batch_X_LL, batch_X_LH, batch_X_HL, batch_X_HH, batch_Y = [], [], [], [], []
        for filename in imageFiles:
            # Complete file path
            file_path = join(imagePath, filename)
            # Load image using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue  # Skip files that didn't load correctly

            # Example of image processing (dummy wavelet transformation)
            transformed_image = np.fft.fft2(image)  # Placeholder for actual wavelet transform
            X_LL, X_LH, X_HL, X_HH = transformed_image.real, transformed_image.imag, transformed_image.real, transformed_image.imag

            # Collect data for the batch
            batch_X_LL.append(X_LL.flatten())
            batch_X_LH.append(X_LH.flatten())
            batch_X_HL.append(X_HL.flatten())
            batch_X_HH.append(X_HH.flatten())
            batch_Y.append(label)

            if len(batch_Y) == batch_size:
                yield [np.array(batch_X_LL), np.array(batch_X_LH), np.array(batch_X_HL), np.array(batch_X_HH)], np.array(batch_Y)
                batch_X_LL, batch_X_LH, batch_X_HL, batch_X_HH, batch_Y = [], [], [], [], []

        # Yield any remaining data as the last batch
        if batch_Y:
            yield [np.array(batch_X_LL), np.array(batch_X_LH), np.array(batch_X_HL), np.array(batch_X_HH)], np.array(batch_Y)

    return data_generator()


# # Usage readWaveletData
# WIDTH, HEIGHT = 128, 128  # Example dimensions, set these to your actual values
# data_gen, total_samples = readWaveletData('path/to/positive/images', 'path/to/negative/images',
#                                           'path/to/positive/training/images', 'path/to/negative/training/images')
# for data in data_gen():
#     X_LL, X_LH, X_HL, X_HH, Y = data
#     print(X_LL, X_LH, X_HL, X_HH, Y)  # Process each sample as needed


# Here, we perform index based splitting and use those indices to split our multi-input datasets. This is done because the CNN model is multi-input network
def splitTrainTestDataForBands(inputData, X_train_ind, X_test_ind):
    X_train = np.zeros((len(X_train_ind), WIDTH*HEIGHT))
    for i in range(len(X_train_ind)):
        X_train[i,:] = inputData[int(X_train_ind[i,0]),:]

    X_test = np.zeros((len(X_test_ind), WIDTH*HEIGHT))
    for i in range(len(X_test_ind)):
        X_test[i,:] = inputData[int(X_test_ind[i,0]),:]

    return X_train, X_test


def countPositiveSamplesAfterSplit(trainData):
    count = 0;
    for i in range(len(trainData)):
        if(trainData[i,0] == 0):
            count = count + 1
    return count


def trainTestSplit(X_LL, X_LH, X_HL, X_HH, X_index, Y, imageCount):
    testCountPercent = 0.1

    # evaluate the model by splitting into train and test sets
    X_train_ind, X_test_ind, Y_train, Y_test = train_test_split(X_index, Y, test_size=testCountPercent, random_state=1, stratify=Y)

    X_LL_train, X_LL_test = splitTrainTestDataForBands(X_LL, X_train_ind, X_test_ind)
    X_LH_train, X_LH_test = splitTrainTestDataForBands(X_LH, X_train_ind, X_test_ind)
    X_HL_train, X_HL_test = splitTrainTestDataForBands(X_HL, X_train_ind, X_test_ind)
    X_HH_train, X_HH_test = splitTrainTestDataForBands(X_HH, X_train_ind, X_test_ind)

    imageHeight = HEIGHT
    imageWidth = WIDTH


    print(countPositiveSamplesAfterSplit(Y_train))
    print(len(X_LL_train))
    print(len(Y_train))
    print(len(X_LL_test))
    print(len(Y_test))

    num_train_samples = len(Y_train)
    print('num_train_samples', num_train_samples)
    X_LL_train = np.array(X_LL_train)
    X_LL_train = X_LL_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_LL_test = np.array(X_LL_test)
    X_LL_test = X_LL_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    X_LH_train = np.array(X_LH_train)
    X_LH_train = X_LH_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_LH_test = np.array(X_LH_test)
    X_LH_test = X_LH_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    X_HL_train = np.array(X_HL_train)
    X_HL_train = X_HL_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_HL_test = np.array(X_HL_test)
    X_HL_test = X_HL_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    X_HH_train = np.array(X_HH_train)
    X_HH_train = X_HH_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_HH_test = np.array(X_HH_test)
    X_HH_test = X_HH_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)


    num_train, height, width, depth = X_LL_train.shape
    num_test = X_LL_test.shape[0]
    num_classes = len(np.unique(Y_train))


    return X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train, X_LL_test, X_LH_test, X_HL_test, X_HH_test, Y_test, height, width, depth, num_classes


# Configuração da Estratégia de Múltiplas GPUs
def trainCNNModel(X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train, X_LL_test, X_LH_test, X_HL_test, X_HH_test,
                Y_test, height, width, depth, num_classes, num_epochs):

    strategy = tf.distribute.MirroredStrategy()  # Define the strategy for multiple GPUs
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    batch_size = 32 * strategy.num_replicas_in_sync  # Adjust batch size according to the number of GPUs

    with strategy.scope():  # Apply the distribution strategy
        model = createModel(height, width, depth, num_classes)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkPointFolder = 'checkPoint'
    checkpoint_name = checkPointFolder + '/Weights-{epoch:03d}--{val_loss:.5f}.keras'

    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks_list = [checkpoint, early_stopping]

    if not os.path.exists(checkPointFolder):
        os.makedirs(checkPointFolder)

    # Create data generators
    train_generator = DataGenerator(X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train, batch_size)
    test_generator = DataGenerator(X_LL_test, X_LH_test, X_HL_test, X_HH_test, Y_test, batch_size)

    # Model training
    model.fit(train_generator, epochs=num_epochs, verbose=1, validation_data=test_generator, callbacks=callbacks_list)

    score, acc = model.evaluate(test_generator, verbose=1)

    model.save('moirePattern3CNN_.keras')

    return model

# Exemplo de uso:
# train_steps = 200  # número de batches de treinamento
# val_steps = 50     # número de batches de validação
# test_steps = 30    # número de batches de teste
# model = trainCNNModel(train_gen, train_steps, val_gen, val_steps, test_gen, test_steps, 128, 128, 4, 2, 10)


# Assumindo que os geradores estejam definidos e que o modelo 'createModel' exista




def evaluate(model, test_gen):
    TP = 0  # True Positives
    TN = 0  # True Negatives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    total_samples = 0  # Total number of samples processed

    for X_test_batch, Y_test_batch in test_gen:
        model_out = model.predict(X_test_batch)
        predictions = np.argmax(model_out, axis=1)

        for i in range(len(Y_test_batch)):
            true_label = Y_test_batch[i]
            predicted_label = predictions[i]
            if true_label == predicted_label:
                if true_label == 0:
                    TP += 1
                else:
                    TN += 1
            else:
                if true_label == 0:
                    FN += 1
                else:
                    FP += 1
        total_samples += len(Y_test_batch)

    # Calculating metrics
    accuracy = (TP + TN) / total_samples
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Output the results
    start = "\033[1m"
    end = "\033[0;0m"
    print(start + 'Confusion matrix (test/validation):' + end)
    print(start + 'True Positive:  ' + end + str(TP))
    print(start + 'False Positive: ' + end + str(FP))
    print(start + 'True Negative:  ' + end + str(TN))
    print(start + 'False Negative: ' + end + str(FN))
    print(start + 'Accuracy:       ' + end + "{:.4f} %".format(100 * accuracy))
    print(start + 'Precision:      ' + end + "{:.4f} %".format(100 * precision))
    print(start + 'Recall:         ' + end + "{:.4f} %".format(100 * recall))

# Assumindo que 'test_gen' é um gerador que produz lotes de dados e etiquetas.


# Main
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('positiveImages', type=str, help='Directory with original positive (Moiré pattern) images.')
    parser.add_argument('negativeImages', type=str, help='Directory with original negative (Normal) images.')

    parser.add_argument('trainingDataPositive', type=str, help='Directory with transformed positive (Moiré pattern) images.')
    parser.add_argument('trainingDataNegative', type=str, help='Directory with transformed negative (Normal) images.')

    parser.add_argument('epochs', type=int, help='Number of epochs for training')

    return parser.parse_args(argv)


# - read positive and negative training data
# - create X and Y from training data

def main(args):
    positiveImagePath = args.positiveImages
    negativeImagePath = args.negativeImages
    numEpochs = args.epochs
    positiveTrainImagePath = args.trainingDataPositive
    negativeTrainImagePath = args.trainingDataNegative

    # Os caminhos para as imagens de treinamento e teste devem ser fornecidos para os geradores
    train_gen = readWaveletData(positiveTrainImagePath, negativeTrainImagePath, mode='train')
    val_gen = readWaveletData(positiveTrainImagePath, negativeTrainImagePath, mode='validation')
    test_gen = readWaveletData(positiveImagePath, negativeImagePath, mode='test')

    # As dimensões dos dados devem ser configuradas corretamente
    height, width = WIDTH, HEIGHT  # Substitua por suas dimensões reais
    depth = 4  # Número de canais (presumindo que cada componente da wavelet é um canal)
    num_classes = 2  # Número de classes (0 para positivo, 1 para negativo)

    # Supondo que trainCNNModel e evaluate são suas funções que treinam e avaliam o modelo
    model = trainCNNModel(train_gen, val_gen, test_gen, height, width, depth, num_classes, numEpochs)

    evaluate(model, test_gen)
    return


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())
    main(parse_arguments(sys.argv[1:]))


    # Chamada à função com os caminhos corretos
    data_gen, total_samples = readWaveletData(positiveImagePath, negativeImagePath)