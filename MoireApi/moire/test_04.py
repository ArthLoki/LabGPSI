import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import logging

import cv2

WIDTH = 750
HEIGHT = 1000


def load_and_evaluate(model_path, test_dir):
    logging.basicConfig(level=logging.INFO)

    # Carrega o modelo completo
    logging.info("Carregando o modelo...")
    model = load_model(model_path)

    # Adjust the number of output channels in a specific layer
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            # Check if the number of output channels is divisible by the number of groups
            print(layer.name, layer.groups, layer.filters, layer.kernel_size, layer.strides)
            if layer.filters % layer.groups != 0:
                # Adjust the number of output channels to make it divisible
                new_filters = layer.filters + (layer.groups - (layer.filters % layer.groups))
                layer.filters = new_filters
                print(f"Adjusted the number of filters in layer {layer.name} to {new_filters}")

    logging.info("Modelo carregado com sucesso.")

    test_data = []
    for e in os.listdir(test_dir):
        img = cv2.imread(os.path.join(test_dir, e))
        test_data.append(img)

    test_data = np.array(test_data, dtype="float") / 255.0

    # Prepara o gerador de dados de teste, agora para imagens em escala de cinza
    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow(test_data, batch_size=1)

    # Avalia o modelo usando o conjunto de dados de teste
    logging.info("Avaliando o modelo...")
    results = model.evaluate(test_generator)
    logging.info(f"Loss: {results[0]}, Accuracy: {results[1]}")

    # Faz previsões sobre todo o conjunto de teste
    logging.info("Gerando previsões para o conjunto de teste...")
    test_generator.reset()
    predictions = model.predict(test_generator, batch_size=32)
    # Reduz as previsões a 1D para binário
    predictions = np.round(predictions).astype(int).flatten()

    # Obtém as etiquetas verdadeiras do conjunto de teste
    true_classes = test_generator.classes

    # Calcula a matriz de confusão
    logging.info("Calculando a matriz de confusão...")
    cm = confusion_matrix(true_classes, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("[[True Negative, False Positive]")
    print(" [False Negative, True Positive]]")

    # Gera relatório de classificação
    logging.info("Gerando relatório de classificação...")
    class_report = classification_report(true_classes,
                                       predictions,
                                       target_names=['Negative', 'Positive'])
    print("Classification Report:")
    print(class_report)

    return {
        "loss": results[0],
        "accuracy": results[1],
        'confusion_matrix': cm,
        'classification_report': class_report
    }
