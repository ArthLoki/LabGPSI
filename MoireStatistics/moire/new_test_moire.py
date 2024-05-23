import os, sys
import numpy as np
from PIL import Image
from moire.mCNN import createModel  # Importando a função de criação do modelo
from keras.models import load_model
from moire.createTrainingData import fwdHaarDWT2D_pywt, normalize_and_convert_image  # Importando funções do script de treinamento

from moire.auxiliary_functions import getCM, count_boolean_channels


# def load_trained_model(weights_path):
#     model = createModel(375, 500, 1, 1)  # Assumindo que estas são as dimensões esperadas pelo modelo
#     model.load_weights(weights_path)
#     return model


def process_image(image_path):
    # Carregar a imagem original e prepará-la
    img = Image.open(image_path)
    img = img.resize((500, 375), Image.Resampling.LANCZOS)  # Redimensionamento para combinar com o treinamento
    img_gray = img.convert('L')
    img_gray = np.array(img_gray, dtype=float)

    # Aplicar a transformada de Haar
    LL, LH, HL, HH = fwdHaarDWT2D_pywt(img_gray)

    # Normalizar os componentes
    LL = normalize_and_convert_image(LL).resize((500, 375), Image.Resampling.LANCZOS)
    LH = normalize_and_convert_image(LH).resize((500, 375), Image.Resampling.LANCZOS)
    HL = normalize_and_convert_image(HL).resize((500, 375), Image.Resampling.LANCZOS)
    HH = normalize_and_convert_image(HH).resize((500, 375), Image.Resampling.LANCZOS)

    # Converter de volta para array numpy para alimentar no modelo
    LL = np.array(LL, dtype=float).reshape(1, 375, 500, 1)
    LH = np.array(LH, dtype=float).reshape(1, 375, 500, 1)
    HL = np.array(HL, dtype=float).reshape(1, 375, 500, 1)
    HH = np.array(HH, dtype=float).reshape(1, 375, 500, 1)

    # print(len(LL), len(LL[0]), len(LL[0][0]),len(LL[0][0][0]))
    # print(len(LH), len(LH[0]), len(LH[0][0]),len(LH[0][0][0]))
    # print(len(HL), len(HL[0]), len(HL[0][0]),len(HL[0][0][0]))
    # print(len(HH), len(HH[0]), len(HH[0][0]),len(HH[0][0][0]))

    # # Carregar o modelo
    # # model = load_trained_model(model_path)
    #
    # # Fazer a predição usando o modelo
    # prediction = model.predict([LL, LH, HL, HH])
    # print('\n\nprediction: ', prediction, '\n\n')
    #
    # # Verificar se a imagem contém o padrão de moiré
    # if prediction[0][0] > 0.5:
    #     print("\nA imagem contém padrão de moiré.\n")
    # else:
    #     print("\nA imagem não contém padrão de moiré.\n")

    return [LL, LH, HL, HH]


# def classify_image(model, image_path):
#     try:
#         # Processa a imagem
#         channels_processed = process_image(image_path)
#         suffixes = ['LL', 'LH', 'HL', 'HH']
#         results = {}
#
#         # Classifica cada componente
#         y_true = []  # Ground truth labels
#         y_pred = []  # Predicted labels
#         for i, suffix in enumerate(suffixes):
#             prediction = model.predict(channels_processed[i])
#             results[suffix] = prediction[0][0] > 0.5  # Assume saída binária
#
#             # Append ground truth and predicted labels
#             y_true.append(1)
#             y_pred.append(1 if results[suffix] else 0)
#
#         print('results: ', results)
#         results_cm = getCM(y_true, y_pred)
#
#         return {'results': results, 'results_cm': results_cm}
#     except Exception as e:
#         return {'Error': 'classify_image error: ' + str(e)}


def classify_imageV2(model, image_path):
    try:
        # Processa a imagem
        components = process_image(image_path)
        suffixes = ['LL', 'LH', 'HL', 'HH']
        results = {}

        # Classifica cada componente
        y_true = []  # Ground truth labels
        y_pred = []  # Predicted labels
        for suffix, component in zip(suffixes, components):
            prediction = model.predict(components)  # [component, component[0], component[0][0], component[0][0][0]]
            results[suffix] = prediction[0][0] > 0.5  # Assume saída binária

            # Append ground truth and predicted labels
            y_true.append(1)
            y_pred.append(1 if results[suffix] else 0)

        print('results: ', results)
        results_cm = getCM(y_true, y_pred)

        return {'results': results, 'results_cm': results_cm}
    except Exception as e:
        return {'Error': 'classify_image error: ' + str(e)}


def load_and_evaluate(model, test_dir, image_name):
    try:
        # # Carrega o modelo
        # model = load_model(model_path)

        # Classifica a imagem
        dict_results = classify_imageV2(model, os.path.join(test_dir, image_name))
        print('dict_results: ', dict_results)

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

