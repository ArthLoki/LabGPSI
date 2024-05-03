import os
import sys
import numpy as np
from PIL import Image
import argparse
from keras.models import load_model
import moire.haar2D as haar2D  # Importa o módulo com a função de transformada de Haar
from sklearn.metrics import confusion_matrix

def normalize_and_resize(component, model_input_shape):
    try:
        # Normaliza o componente para o intervalo [0, 1] e redimensiona conforme o modelo espera
        component_normalized = (component - np.min(component)) / (np.max(component) - np.min(component))
        component_resized = np.resize(component_normalized, (1, model_input_shape[0], model_input_shape[1], 1))
        return component_resized
    except Exception as e:
        return {'Error': 'normalize_and_resize error: ' + str(e)}

def classify_image(model, image_path):
    try:
        # Carrega e prepara a imagem
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=float)

        # Aplica a transformada de Haar
        cA, cH, cV, cD = haar2D.fwdHaarDWT2D(img_array)
        components = [cA, cH, cV, cD]
        suffixes = ['LL', 'LH', 'HL', 'HH']
        results = {}

        # Classifica cada componente
        for suffix, component in zip(suffixes, components):
            component_ready = normalize_and_resize(component, model.input_shape[1:3])
            prediction = model.predict(component_ready)
            results[suffix] = prediction[0][0] > 0.5  # Assume saída binária

        return results
    except Exception as e:
        return {'Error': 'classify_image error: ' + str(e)}


def load_and_evaluate(model_path, test_dir):

    try:
        # Carrega o modelo
        model = load_model(model_path)

        # Classifica a imagem
        results = classify_image(model, os.path.join(test_dir, os.listdir(test_dir)[0]))

        # Verifica e imprime os resultados
        has_moire = any(results.values())
        channels_with_moire = [key for key, value in results.items() if value]
        print(f"A imagem contém moiré? {'Sim' if has_moire else 'Não'}")
        if has_moire:
            print(f"Canais com detecção de moiré: {', '.join(channels_with_moire)}")

        return {'moire': has_moire}
    except Exception as e:
        return {'Error': 'load_and_evaluate error: ' + str(e)}
