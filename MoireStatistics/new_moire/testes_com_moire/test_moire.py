import sys
import numpy as np
from PIL import Image
from mCNN import createModel  # Importando a função de criação do modelo
from createTrainingData import fwdHaarDWT2D_pywt, normalize_and_convert_image  # Importando funções do script de treinamento
from keras.models import load_model

def load_trained_model(weights_path):
    model = createModel(375, 500, 1, 1)  # Assumindo que estas são as dimensões esperadas pelo modelo
    model.load_weights(weights_path)
    return model

def process_and_classify_image(image_path, model_path):
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

    # Carregar o modelo
    model = load_trained_model(model_path)

    # Fazer a predição usando o modelo
    prediction = model.predict([LL, LH, HL, HH])

    # Verificar se a imagem contém o padrão de moiré
    if prediction[0][0] > 0.5:
        print("A imagem contém padrão de moiré.")
    else:
        print("A imagem não contém padrão de moiré.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Por favor, forneça o caminho da imagem e o caminho do modelo.")
    else:
        process_and_classify_image(sys.argv[1], sys.argv[2])
