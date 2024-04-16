import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import logging

WIDTH = 750
HEIGHT = 1000


def load_and_evaluate(model_path, test_dir):
    logging.basicConfig(level=logging.INFO)

    # Carrega o modelo completo
    logging.info("Carregando o modelo...")
    model = load_model(model_path)
    logging.info("Modelo carregado com sucesso.")

    # print("teste_dir_image:", os.listdir(test_dir))

    # Prepara o gerador de dados de teste, agora para imagens em escala de cinza
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # print("test_datagen:", test_datagen)
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(HEIGHT, WIDTH),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale',  # Modificado para escala de cinza
        shuffle=False  # Não misturar os dados para correspondência de previsões
    )

    # print("test_generator:", test_generator)
    # print("test_generator.classes:", test_generator.classes)

    # Calcula o número correto de passos por época
    steps_per_epoch = np.ceil(test_generator.samples / test_generator.batch_size)
    # print('test_generator.samples:', test_generator.samples, 'test_generator.batch_size:', test_generator.batch_size)

    if steps_per_epoch == 0.0:
        steps_per_epoch = 1.0

    # Avalia o modelo usando o conjunto de dados de teste
    logging.info("Avaliando o modelo...")
    results = model.evaluate(test_generator, steps=steps_per_epoch)
    logging.info(f"Loss: {results[0]}, Accuracy: {results[1]}")

    # Faz previsões sobre todo o conjunto de teste
    logging.info("Gerando previsões para o conjunto de teste...")
    test_generator.reset()
    predictions = model.predict(test_generator, steps=steps_per_epoch)
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
