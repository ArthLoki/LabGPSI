import os, sys
import numpy as np
from PIL import Image
from moire.mCNN import createModel  # Importando a função de criação do modelo
from keras.models import load_model
from moire.createTrainingData import fwdHaarDWT2D_pywt, normalize_and_convert_image  # Importando funções do script de treinamento

from moire.auxiliary_functions import getCM, count_boolean_channels


def load_trained_model(weights_path):
    model = createModel(375, 500, 1, 1)  # Assumindo que estas são as dimensões esperadas pelo modelo
    model.load_weights(weights_path)
    return model


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


def classify_image(model, image_path):
    try:
        # Processa a imagem
        components = process_image(image_path)

        # Classifica componentes
        prediction = model.predict(components)

        print('prediction[0]: ', prediction[0])
        return {'results': True if prediction[0][0] > 0.5 else False}
    except Exception as e:
        return {'Error': 'classify_image error: ' + str(e)}


def load_and_evaluate(model, test_dir, image_name):
    try:
        # Classifica a imagem
        dict_results = classify_image(model, os.path.join(test_dir, image_name))
        print('dict_results: ', dict_results)

        result = dict_results.get('results')
        if result is None:
            return {'Error': 'load_and_evaluate: results cannot be None'}

        # Verifica e imprime os resultados
        print(f"A imagem contém moiré? {'Sim' if result else 'Não'}")

        return {
            'moire': result,
        }
    except Exception as e:
        return {'Error': 'load_and_evaluate error: ' + str(e)}

