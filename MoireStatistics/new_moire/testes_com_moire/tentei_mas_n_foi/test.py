import sys
import os
import numpy as np
from mCNN import createModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image
from train_lotes import load_image 


# Configurações do modelo e das imagens
config = {
    "width": 512,
    "height": 384,
    "depth": 1,
    "num_classes": 2,
    "batch_size": 32,  
}

def load_trained_model(weights_path, config):
    model = createModel(config["height"], config["width"], config["depth"], config["num_classes"]) 
    model.load_weights(weights_path)
    return model





def image_data_predictor(positive_dir, negative_dir, batch_size):
    """ Gera batches de dados com quatro componentes de imagens para predição. """
    # Coleta os arquivos dos diretórios positive e negative
    image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(positive_dir) for f in filenames if f.endswith('.tiff')]
    image_files += [os.path.join(dp, f) for dp, dn, filenames in os.walk(negative_dir) for f in filenames if f.endswith('.tiff')]
    
    num_samples = len(image_files) // 4
    for offset in range(0, num_samples, batch_size):
        # Seleciona um subconjunto de arquivos de acordo com o tamanho do batch
        batch_files = image_files[offset*4:(offset + batch_size)*4]
        # Ordena os arquivos primeiramente pela raiz do nome, mantendo as variações e sufixos corretos agrupados
        batch_files_sorted = sorted(batch_files, key=lambda x: (x.rsplit('_', 2)[0], x.rsplit('_', 1)[1]))
        X_LL, X_LH, X_HL, X_HH = [], [], [], []
        # Verifica se o número de arquivos no batch é adequado para evitar lote incompleto
        if len(batch_files_sorted) % 4 != 0:
            continue  # Evita processar lotes incompletos
        
        # Certifica que cada grupo de 4 arquivos corresponde aos canais LL, LH, HL, HH respectivamente
        for i in range(0, len(batch_files_sorted), 4):
            # Verifica se os componentes estão na ordem correta
            components = ['LL', 'LH', 'HL', 'HH']
            if all(batch_files_sorted[j].endswith(components[j]) for j in range(4)):
                X_LL.append(load_image(batch_files_sorted[i]))
                X_LH.append(load_image(batch_files_sorted[i+1]))
                X_HL.append(load_image(batch_files_sorted[i+2]))
                X_HH.append(load_image(batch_files_sorted[i+3]))
            else:
                continue  # Pula se os componentes não estão na ordem esperada

        # Organiza os canais de cada imagem e retorna um array
        yield np.stack([X_LL, X_LH, X_HL, X_HH], axis=-1)


def process_directory(positive_dir, negative_dir, model):
    labels = []
    predictions = []

    # Processa as imagens no diretório 'positive'
    for filename in os.listdir(positive_dir):
        image_path = os.path.join(positive_dir, filename)
        pred = process_and_classify_image(image_path, model)
        predictions.append(pred)
        labels.append(1)  # Etiqueta para imagens positivas

    # Processa as imagens no diretório 'negative'
    for filename in os.listdir(negative_dir):
        image_path = os.path.join(negative_dir, filename)
        pred = process_and_classify_image(image_path, model)
        predictions.append(pred)
        labels.append(0)  # Etiqueta para imagens negativas

    return labels, predictions



def process_and_classify_image(component_paths, model):
    # A função supõe que 'component_paths' é uma lista com caminhos para as imagens dos componentes LL, LH, HL, HH
    # Carrega os componentes diretamente como arrays de ponto flutuante e normaliza-os
    components = [np.array(Image.open(path), dtype=np.float32) / 255.0 for path in component_paths]

    # Assumindo que cada componente foi normalizado e está pronto para ser empilhado como entrada
    inputs = np.stack(components, axis=-1).reshape(1, 375, 500, 4)  # Reformatar para o formato de entrada esperado pelo modelo

    # Fazer a predição
    prediction = model.predict(inputs)

    # Retorna se a imagem é classificada como contendo padrão de moiré (1) ou não (0)
    return 1 if prediction[0][0] > 0.5 else 0


def print_and_save_confusion_matrix(cm, true_labels, predicted_labels, file_name):
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
    if len(sys.argv) < 4:
        print("Por favor, forneça os diretórios das imagens positivas e negativas e o caminho do modelo.")
    else:
        positive_dir = sys.argv[1]
        negative_dir = sys.argv[2]
        model_path = sys.argv[3]
        model = load_trained_model(model_path, config)
        true_labels, predicted_labels = process_directory(positive_dir, negative_dir, model)
        cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])
        print_and_save_confusion_matrix(cm, true_labels, predicted_labels, 'confusion_matrix.txt')
        print("Matriz de confusão e métricas salvas em 'confusion_matrix.txt'.")
