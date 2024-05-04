import os
import numpy as np
from PIL import Image
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


def getCM(y_true, y_pred, count_channels):
    try:
        # Compute confusion matrix
        '''
        Confusion Matrix:

        [[true negative, false positive],
        [false negative, true positive]]
        '''

        confusion_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Check Confusion Matrix data
        TN = confusion_mat[0, 0]
        FP = confusion_mat[0, 1]
        FN = confusion_mat[1, 0]
        TP = confusion_mat[1, 1]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results_cm = {
            # 'Confusion Matrix': confusion_mat,
            "True Positive": TP,
            "True Negative": TN,
            "False Positive": FP,
            "False Negative": FN,
            "Accuracy": accuracy,
            'Precision': precision,
            'Recall (Sensitivity)': recall,
            'F1 Score': f1score,
        }

        return results_cm
    except Exception as e:
        return {'Error': 'getCM error: ' + str(e)}


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
        y_true = []  # Ground truth labels
        y_pred = []  # Predicted labels
        for suffix, component in zip(suffixes, components):
            component_ready = normalize_and_resize(component, model.input_shape[1:3])
            prediction = model.predict(component_ready)
            results[suffix] = prediction[0][0] > 0.5  # Assume saída binária

            # Append ground truth and predicted labels
            y_true.append(1)
            y_pred.append(1 if results[suffix] else 0)

        print('results: ', results)
        results_cm = getCM(y_true, y_pred, len(suffixes))

        return {'results': results, 'results_cm': results_cm}
    except Exception as e:
        return {'Error': 'classify_image error: ' + str(e)}


def count_boolean_channels(results):
    # Create counter variable
    channels_boolean_counter = {
        "LL": {True: 0, False: 0},
        "LH": {True: 0, False: 0},
        "HL": {True: 0, False: 0},
        "HH": {True: 0, False: 0},
    }

    # Increment counter variable
    for key, value in results.items():
        channels_boolean_counter[key][value] += 1

    return channels_boolean_counter


def load_and_evaluate(model_path, test_dir, image_name):
    try:
        # Carrega o modelo
        model = load_model(model_path)

        # Classifica a imagem
        dict_results = classify_image(model, os.path.join(test_dir, image_name))

        results, results_cm = dict_results.get('results'), dict_results.get('results_cm')

        if results is None:
            return {'Error': 'load_and_evaluate: results cannot be None'}

        channels_boolean_counter = count_boolean_channels(results)

        if results_cm is None:
            return {'Error': 'load_and_evaluate: results_cm cannot be None'}

        # Verifica e imprime os resultados
        has_moire = any(results.values())
        channels_with_moire = [key for key, value in results.items() if value]
        print(f"A imagem contém moiré? {'Sim' if has_moire else 'Não'}")
        if has_moire:
            print(f"Canais com detecção de moiré: {', '.join(channels_with_moire)}")

        return {
            'results_channels': channels_boolean_counter,
            'results_cm': results_cm,
            'moire': has_moire,
        }
    except Exception as e:
        return {'Error': 'load_and_evaluate error: ' + str(e)}
