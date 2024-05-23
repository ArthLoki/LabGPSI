import sys
import os
import numpy as np
from PIL import Image
from mCNN import createModel
from createTrainingData import fwdHaarDWT2D_pywt, normalize_and_convert_image
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_trained_model(weights_path):
    model = createModel(375, 500, 1, 1)  # Assumindo as dimensões esperadas pelo modelo
    model.load_weights(weights_path)
    return model

def process_and_classify_image(image_path, model):
    # Carregar e preparar a imagem
    img = Image.open(image_path).resize((500, 375), Image.Resampling.LANCZOS).convert('L')
    img = np.array(img, dtype=float)

    # Aplicar transformada de Haar e normalizar
    components = fwdHaarDWT2D_pywt(img)
    processed = [normalize_and_convert_image(c).resize((500, 375), Image.Resampling.LANCZOS) for c in components]
    inputs = [np.array(c, dtype=float).reshape(1, 375, 500, 1) for c in processed]

    # Fazer predição
    prediction = model.predict(inputs)

    # Retorna se a imagem é classificada como contendo padrão de moiré (1) ou não (0)
    return 1 if prediction[0][0] > 0.5 else 0

def process_directory(directory, model):
    labels = []
    predictions = []
    for category in ['positive', 'negative']:
        folder = os.path.join(directory, category)
        for filename in os.listdir(folder):
            image_path = os.path.join(folder, filename)
            pred = process_and_classify_image(image_path, model)
            predictions.append(pred)
            labels.append(1 if category == 'positive' else 0)
    return labels, predictions

def print_and_save_confusion_matrix(cm, file_name):
    # Detalhes para imprimir a matriz de confusão de forma mais clara
    print("Matriz de Confusão:")
    print(f"Verdadeiro Positivo (VP): {cm[0][0]}")
    print(f"Falso Negativo (FN): {cm[0][1]}")
    print(f"Falso Positivo (FP): {cm[1][0]}")
    print(f"Verdadeiro Negativo (VN): {cm[1][1]}")

    # Calculando métricas
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Imprimindo métricas
    print(f"Precisão: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Medida F1: {f1:.2f}")

    # Salva a matriz de confusão em um arquivo de texto
    with open(file_name, 'w') as f:
        f.write("Matriz de Confusão:\n")
        f.write(f"VP: {cm[0][0]} FN: {cm[0][1]}\n")
        f.write(f"FP: {cm[1][0]} VN: {cm[1][1]}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Por favor, forneça o diretório das imagens e o caminho do modelo.")
    else:
        model = load_trained_model(sys.argv[2])
        true_labels, predicted_labels = process_directory(sys.argv[1], model)
        cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])
        print_and_save_confusion_matrix(cm, 'confusion_matrix.txt')
        print("Matriz de confusão e métricas salvas em 'confusion_matrix.txt'.")
