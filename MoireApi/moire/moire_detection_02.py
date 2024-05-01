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

        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else (TP + TN) / count_channels
        precision = TP / (TP + FP) if TP + FP > 0 else TP
        recall = TP / (TP + FN) if TP + FN > 0 else TP
        f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 2 * (
                    precision * recall)

        # is_moire = 'under analysis'
        # if TN + FP + FN + TP == count_channels:
        #     if accuracy > 0.7:
        #         # In this case, to make sure no false negative passes, we will consider them as false positives
        #         # if ((TN == TRANSFORM_COUNT) and (FP+TP+FN == 0)) or (0 < TP < TN and TN > 0): is_moire = 'negative'
        #         # elif ((TP == TRANSFORM_COUNT) and (FP+TN+FN == 0)) or (0 < TN < TP and TP > 0): is_moire = 'positive'
        #         # else: is_moire = 'false positive'
        #
        #         if (TN == count_channels) and (FP + TP + FN == 0):
        #             is_moire = 'negative'
        #         elif (TP == count_channels) and (FP + TN + FN == 0):
        #             is_moire = 'positive'
        #         else:
        #             is_moire = 'false positive'
        #     else:
        #         # If accuracy is not good, everything becomes 'false positive' for manual analysis
        #         is_moire = 'false positive'

        results_cm = {
            # 'Confusion Matrix': confusion_mat,
            # 'Moire': is_moire,
            "True Positives": TP,
            "True Negatives": TN,
            "False Positives": FP,
            "False Negatives": FN,
            "Accuracy": accuracy,
            'Precision': precision,
            'Recall': recall,
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
        results_pred = []  # Results from prediction
        y_true = []  # Ground truth labels
        y_pred = []  # Predicted labels
        for suffix, component in zip(suffixes, components):
            component_ready = normalize_and_resize(component, model.input_shape[1:3])
            prediction = model.predict(component_ready)
            results[suffix] = prediction[0][0] > 0.5  # Assume saída binária

            results_pred.append({'prediction': prediction[0][0]})

            # Append ground truth and predicted labels
            y_true.append(1)
            y_pred.append(1 if results[suffix] else 0)

        print('results: ', results)
        results_cm = getCM(y_true, y_pred, len(suffixes))

        return {'results': results, 'results_predictions': results_pred, 'results_cm': results_cm}
    except Exception as e:
        return {'Error': 'classify_image error: ' + str(e)}


def load_and_evaluate(model_path, test_dir):
    try:
        # Carrega o modelo
        model = load_model(model_path)

        # Classifica a imagem
        dict_results = classify_image(model, os.path.join(test_dir, os.listdir(test_dir)[0]))

        results, results_pred, results_cm = dict_results.get('results'), dict_results.get(
            'results_predictions'), dict_results.get('results_cm')

        if results is None:
            return {'Error': 'load_and_evaluate: results cannot be None'}

        if results_pred is None:
            return {'Error': 'load_and_evaluate: results_pred cannot be None'}

        if results_cm is None:
            return {'Error': 'load_and_evaluate: results_cm cannot be None'}

        # Verifica e imprime os resultados
        has_moire = any(results.values())
        channels_with_moire = [key for key, value in results.items() if value]
        print(f"A imagem contém moiré? {'Sim' if has_moire else 'Não'}")
        if has_moire:
            print(f"Canais com detecção de moiré: {', '.join(channels_with_moire)}")

        return {
            'results': results,
            'results_predictions': results_pred,
            'results_cm': results_cm,
            'moire': has_moire,
        }
    except Exception as e:
        return {'Error': 'load_and_evaluate error: ' + str(e)}
