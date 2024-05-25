from tensorflow.keras.utils import Sequence
import numpy as np

# Nova classe para gerar lotes de dados
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
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.imageFiles) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.imageFiles[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_files, batch_labels)
        return X, y

    def on_epoch_end(self):
        combined = list(zip(self.imageFiles, self.labels))
        np.random.shuffle(combined)
        self.imageFiles[:], self.labels[:] = zip(*combined)

    def __data_generation(self, batch_files, batch_labels):
        X_LL = np.empty((self.batch_size, self.height * self.width))
        X_LH = np.empty((self.batch_size, self.height * self.width))
        X_HL = np.empty((self.batch_size, self.height * self.width))
        X_HH = np.empty((self.batch_size, self.height * self.width))
        Y = np.empty(self.batch_size, dtype=int)

        for i, (f, label) in enumerate(zip(batch_files, batch_labels)):
            imagePath = self.positiveImagePath if label == 0 else self.negativeImagePath
            readAndScaleImage(f, '', imagePath, X_LL, X_LH, X_HL, X_HH, np.zeros((self.batch_size, 1)), Y, i, label)

        X = [X_LL.reshape(self.batch_size, self.height, self.width, 1),
             X_LH.reshape(self.batch_size, self.height, self.width, 1),
             X_HL.reshape(self.batch_size, self.height, self.width, 1),
             X_HH.reshape(self.batch_size, self.height, self.width, 1)]

        return X, to_categorical(Y, num_classes=2)