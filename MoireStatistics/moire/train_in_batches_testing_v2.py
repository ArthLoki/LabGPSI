# 0 - Import Dependencies
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

# from DataGenerator import DataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow import TensorShape


# Global constants
WIDTH = 500#384
HEIGHT = 375#512

'''
Function order according to its use:
    1 - countPositiveSamplesAfterSplit:
        >> Esta função conta o número de amostras positivas (ou seja, rótulo 0) no conjunto de dados de 
        treinamento após a divisão do conjunto de treinamento. Isso pode ser útil para entender a 
        distribuição das classes no conjunto de dados e avaliar o desempenho de algoritmos de classificação.

    2 - splitTrainTestDataForBands:
        >> Here, we perform index based splitting and use those indices to split our multi-input datasets.
        This is done because the CNN model is multi-input network.
        
        >> É responsável por dividir os dados de entrada (normalmente um conjunto de dados de entrada) em 
        conjuntos de treinamento e teste com base em índices fornecidos

    3 - scaleData:
        >> É responsável por aplicar uma transformação de escala aos dados de entrada utilizando 
        a técnica de escala mínima-máxima (MinMaxScaler).
        
        >> Usada para garantir que os dados estejam em uma faixa específica antes de serem utilizados em algoritmos 
        de aprendizado de máquina, onde a escala dos dados pode afetar o desempenho do modelo.

    4 - readImageSet:
        >> É responsável por ler as imagens do conjunto de dados, aplicar uma transformação de escala nelas 
        e armazenar os valores resultantes em matrizes específicas (X_LL, X_LH, X_HL, X_HH, X_index, Y) usadas 
        para treinar um modelo de aprendizado de máquina.
        
        >> Fundamental para preparar os dados de treinamento antes de alimentá-los em um modelo de aprendizado 
        de máquina. Ao ler as imagens, aplicar transformações de escala e manipular as matrizes de dados, ela 
        cria o conjunto de dados que será usado para treinar o modelo.

    5 - readAndScaleImage:
        >> As funções readWaveletData, trainCNNModel, e evaluate devem ser definidas para lidar com geradores
        e configuração apropriada de treinamento, validação e teste

    6 - evaluate:
        >> 

    7 - trainCNNModel:
        >> 
        
        >> Usage example:
            train_steps = 200  # número de batches de treinamento
            val_steps = 50     # número de batches de validação
            test_steps = 30    # número de batches de teste
            model = trainCNNModel(train_gen, train_steps, val_gen, val_steps, test_gen, test_steps, 128, 128, 4, 2, 10)

    8 - readWaveletData:
        >> 
        
        >> Usage example:
            train_gen = readWaveletData('path/to/positive/images', 'path/to/negative/images', mode='train')
            val_gen = readWaveletData('path/to/positive/images', 'path/to/negative/images', mode='validation')
            test_gen = readWaveletData('path/to/positive/images', 'path/to/negative/images', mode='test')

    9 - parse_arguments:
        >>

    10 - main
        >> 
'''


# 1 - countPositiveSamplesAfterSplit
def countPositiveSamplesAfterSplit(trainData):
    count = 0;
    for i in range(len(trainData)):
        if(trainData[i,0] == 0):
            count = count + 1
    return count


# 2 - splitTrainTestDataForBands
def splitTrainTestDataForBands(inputData, X_train_ind, X_test_ind):
    X_train = np.zeros((len(X_train_ind), WIDTH*HEIGHT))
    for i in range(len(X_train_ind)):
        X_train[i,:] = inputData[int(X_train_ind[i,0]),:]

    X_test = np.zeros((len(X_test_ind), WIDTH*HEIGHT))
    for i in range(len(X_test_ind)):
        X_test[i,:] = inputData[int(X_test_ind[i,0]),:]

    return X_train, X_test


# 3 - scaleData
def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum, maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)

    return inp


def scaleDataV2(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum, maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)

    return inp


# 4 - readImageSet
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


# 5 - readAndScaleImage
def readAndScaleImage(f, customStr, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, sampleVal):
    fileName = (os.path.splitext(f)[0])
    # fLL = (f.replace(fileName, fileName + customStr + '_LL')).replace('.jpg', '.tiff')
    # fLH = (f.replace(fileName, fileName + customStr + '_LH')).replace('.jpg', '.tiff')
    # fHL = (f.replace(fileName, fileName + customStr + '_HL')).replace('.jpg', '.tiff')
    # fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg', '.tiff')

    fLL = f if 'LL' in fileName else f
    fLH = f if 'LH' in fileName else f
    fHL = f if 'HL' in fileName else f
    fHH = f if 'HH' in fileName else f

    try:
        imgLL = Image.open(str(join(trainImagePath, fLL)).replace('\\', '/'))
        imgLH = Image.open(str(join(trainImagePath, fLH)).replace('\\', '/'))
        imgHL = Image.open(str(join(trainImagePath, fHL)).replace('\\', '/'))
        imgHH = Image.open(str(join(trainImagePath, fHH)).replace('\\', '/'))
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

    # Assuming Y is initially a 2D array with multiple columns
    Y = Y.reshape(-1, 1)  # Reshape to have all rows and 1 column
    Y[sampleIndex, 0] = sampleVal
    X_index[sampleIndex, 0] = sampleIndex

    return True


# 6 - evaluate
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

    return

# Assumindo que 'test_gen' é um gerador que produz lotes de dados e etiquetas.


# 7 - trainCNNModel
# Modificando a função trainCNNModel para utilizar os geradores de dados
def trainCNNModel(train_gen, val_gen, test_gen, height, width, depth, num_classes, num_epochs):

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

    # Model training
    model.fit(train_gen, epochs=num_epochs, verbose=1, validation_data=val_gen, callbacks=callbacks_list)

    # Evaluate the model
    score, acc = model.evaluate(test_gen, verbose=1)
    print('\n>> Score: ', score)
    print('\n>> Accuracy: ', acc)

    model.save('moirePattern3CNN_.keras')

    return model

# Assumindo que os geradores estejam definidos e que o modelo 'createModel' exista


# 8 - readWaveletData
# class DataGenerator(Sequence):
#     def __init__(self, imageFiles, labels, batch_size, positiveImagePath, negativeImagePath, height, width, channels):
#         self.imageFiles = imageFiles
#         self.labels = labels
#         self.batch_size = batch_size
#         self.positiveImagePath = positiveImagePath
#         self.negativeImagePath = negativeImagePath
#         self.height = height
#         self.width = width
#         self.channels = channels
#         self.on_epoch_end()
#
#         # Shuffle data for random sampling within batches
#         self.indexes = np.arange(len(imageFiles))
#         np.random.shuffle(self.indexes)
#
#     def __len__(self):
#         return int(np.floor(len(self.imageFiles) / self.batch_size))
#
#     def __getitem__(self, index):
#         batch_files = self.imageFiles[index * self.batch_size:(index + 1) * self.batch_size]
#         batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
#         X, y = self.__data_generation(batch_files, batch_labels)
#         return X, y
#
#     @property
#     def output_signature(self):
#         return {
#             'image': tf.TensorShape([self.height, self.width, self.channels]),
#             'label': tf.TensorShape([1])
#         }
#
#     def on_epoch_end(self):
#         combined = list(zip(self.imageFiles, self.labels))
#         np.random.shuffle(combined)
#         self.imageFiles[:], self.labels[:] = zip(*combined)
#
#     def __data_generation(self, list_IDs_temp):
#         X = np.empty((self.batch_size, self.height, self.width, self.channels))  # Pre-allocate for images
#         y = np.empty(self.batch_size)  # Pre-allocate for labels
#
#         # Loop through each data point
#         for i, ID in enumerate(list_IDs_temp):
#             # Determine image path based on label
#             imagePath = self.positiveImagePath if self.labels[ID] == 0 else self.negativeImagePath
#
#             # Load image data
#             # Replace this with your image loading logic (e.g., OpenCV, scikit-image)
#             image = cv2.imread(imagePath)  # Assuming you're using OpenCV
#
#             # Preprocess image (optional)
#             # You can add resizing, normalization, etc. here
#             image = cv2.resize(image, (self.width, self.height))
#             image = image.astype('float32') / 255.0  # Normalize to 0-1 range
#
#             # Assign image and label data
#             X[i,] = image
#             y[i] = self.labels[ID]
#
#         return X, y

class DataGenerator(Sequence):
    def __init__(self, imageFiles, labels, batch_size, positiveImagePath, negativeImagePath, height, width, channels):
        self.imageFiles = imageFiles
        self.labels = labels
        self.batch_size = batch_size
        self.positiveImagePath = positiveImagePath
        self.negativeImagePath = negativeImagePath
        self.height = height
        self.width = width
        self.channels = channels

        # Shuffle data for random sampling within batches
        self.indexes = np.arange(len(imageFiles))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.imageFiles) / self.batch_size))

    def __getitem__(self, index):
        # Get the list of data point IDs for this batch
        list_IDs_temp = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data and labels for this batch
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_image_path(self, imageFile):
        imagePath = join(self.positiveImagePath, imageFile)
        if not os.path.exists(imagePath):
            imagePath = join(self.negativeImagePath, imageFile)
        return imagePath

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.height, self.width, self.channels))  # Pre-allocate for images
        y = np.empty(self.batch_size)  # Pre-allocate for labels

        # Loop through shuffled data points (using self.indexes)
        for i, ID in enumerate(list_IDs_temp):
            # Access data point based on shuffled index
            index = self.indexes[ID]
            imageFile = self.imageFiles[index]
            label = self.labels[index]

            # Get image path
            imagePath = self.get_image_path(imageFile)

            # Call process_data to load and process the data
            image_data, label = self.process_data(imagePath, label)

            image_data = np.expand_dims(image_data, axis=-1)  # Add dimension at the end (axis=-1)

            # Assign image and label data
            X[i,] = image_data
            y[i] = label

        return X, y

    def process_data(self, imagePath, label):
        try:
            # Load image data
            image = cv2.imread(imagePath)

            if image is None:
                print("Could not open image", imagePath)
                exit(1)

            # RGB to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Reshape to add an extra dimension of size 1 for Keras
            image = np.expand_dims(gray_image, axis=-1)  # Add dimension at the end (axis=-1)
        except Exception as e:
            print("Error while processing data: ", e)
            exit(1)

        # Preprocess image (optional)
        # You can add resizing, normalization, etc. here
        image = cv2.resize(image, (self.width, self.height))
        image = image.astype('float32') / 255.0  # Normalize to 0-1 range

        return image, label

    @property
    def output_signature(self):
        return {
            'image': tf.TensorShape([self.height, self.width, self.channels]),
            'label': tf.TensorShape([1])
        }

# Modificando a função readWaveletData para retornar os geradores de dados
def readWaveletData(positiveImagePath, negativeImagePath, mode='train', batch_size=32):
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
    # np.random.shuffle(imageFiles)

    # Combine positive and negative image files
    imageFiles = positiveImageFiles + negativeImageFiles
    labels = [0] * len(positiveImageFiles) + [1] * len(negativeImageFiles)

    # Shuffle the files and labels together
    combined = list(zip(imageFiles, labels))
    np.random.shuffle(combined)
    imageFiles[:], labels[:] = zip(*combined)

    data_gen = DataGenerator(imageFiles, labels, batch_size, positiveImagePath, negativeImagePath, HEIGHT, WIDTH, 1)

    print('data_gen:', data_gen)

    return data_gen


# 9 - parse_arguments
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('positiveImages', type=str, help='Directory with original positive (Moiré pattern) images.')
    parser.add_argument('negativeImages', type=str, help='Directory with original negative (Normal) images.')

    # parser.add_argument('trainingDataPositive', type=str, help='Directory with transformed positive (Moiré pattern) images.')
    # parser.add_argument('trainingDataNegative', type=str, help='Directory with transformed negative (Normal) images.')

    parser.add_argument('epochs', type=int, help='Number of epochs for training')

    return parser.parse_args(argv)


# 10 - main
def main(args):
    # - read positive and negative training data
    # - create X and Y from training data
    positiveImagePath = args.positiveImages
    negativeImagePath = args.negativeImages

    numEpochs = args.epochs
    # positiveTrainImagePath = args.trainingDataPositive
    # negativeTrainImagePath = args.trainingDataNegative
    positiveTrainImagePath = '{}/trainingDataPositive'.format(str(os.getcwd()).replace("\\", "/"))
    negativeTrainImagePath = '{}/trainingDataNegative'.format(str(os.getcwd()).replace("\\", "/"))

    # os.system(
    #     'python createTrainingData.py --positiveImages ' + positiveImagePath + ' --negativeImages ' + positiveImagePath + ' 0'
    # )

    # Os caminhos para as imagens de treinamento e teste devem ser fornecidos para os geradores
    train_gen = readWaveletData(positiveTrainImagePath, negativeTrainImagePath, mode='train')
    val_gen = readWaveletData(positiveTrainImagePath, negativeTrainImagePath, mode='validation')
    test_gen = readWaveletData(positiveImagePath, negativeImagePath, mode='test')

    # As dimensões dos dados devem ser configuradas corretamente
    height, width = WIDTH, HEIGHT  # Substitua por suas dimensões reais
    depth = 4  # Número de canais (presumindo que cada componente da wavelet é um canal)
    num_classes = 2  # Número de classes (0 para positivo, 1 para negativo)

    print('Number of training images: {}'.format(len(train_gen)))
    print('train_gen: {}'.format(train_gen))

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