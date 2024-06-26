import sys
import os
import numpy as np
from mCNN import createModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image
from train_lotes import load_image 


# Configurações do modelo e das imagens
config = {
    "width": 500,
    "height": 375,
    "depth": 1,
    "num_classes": 2,
    "batch_size": 32,  
}

def load_trained_model(weights_path, config):
    model = createModel(config["height"], config["width"], config["depth"], config["num_classes"]) 
    model.load_weights(weights_path)
    return model


def image_data_predictor(image_files, batch_size):
    """ Gera batches de dados com quatro componentes de imagens para predição. """
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




def process_directories(positive_dir, negative_dir, model):
    labels = []
    predictions = []
    # Combina os diretórios e as categorias correspondentes em uma lista de tuplas
    categories = [(positive_dir, 1), (negative_dir, 0)]
    
    for directory, label in categories:
        print(f"Processando diretório: {directory} com label: {label}")
        # Agrupamento dos caminhos dos componentes por raiz do nome do arquivo
        component_files = {}
        for filename in os.listdir(directory):
            # Isola a raiz do nome do arquivo e o sufixo do componente
            root_name = "_".join(filename.split('_')[:-1])
            print(f"Componentes para {root_name}: {components}")
            if root_name not in component_files:
                component_files[root_name] = []
            component_files[root_name].append(os.path.join(directory, filename))
        
        # Processa cada grupo de componentes
        for root_namo, components in component_files.values():
            if len(components) == 4:  # Garante que todos os 4 componentes estejam presentes
                # Ordena os componentes para garantir a ordem LL, LH, HL, HH
                components_sorted = sorted(components, key=lambda x: x.split('_')[-1].split('.')[0])
                pred = process_and_classify_image(components_sorted, model)
                predictions.append(pred)
                labels.append(label)

    return labels, predictions

def process_and_classify_image(component_paths, model):
    # A função agora espera que 'component_paths' seja uma lista com caminhos para as imagens dos componentes LL, LH, HL, HH
    # Carrega cada componente como um array de ponto flutuante normalizado
    print(f"Classificando imagem com caminhos de componentes: {component_paths}")
    #components = [np.expand_dims(np.array(Image.open(path), dtype=np.float32) / 255.0, axis=-1) for path in component_paths]
    for path in component_paths:
        img = Image.open(path)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        components.append(img_array)
        print(f"Formato do componente {path}: {img_array.shape}")
    # Predição usando o modelo, passando cada componente como uma entrada separada
    prediction = model.predict([components[0], components[1], components[2], components[3]])
    print(f"Predição: {prediction}")
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
        print("Por favor, forneça os diretórios das imagens 'positive' e 'negative' e o caminho do modelo.")
    else:
        print("Carregando modelo...")
        model = load_trained_model(sys.argv[3], config)
        print("Modelo carregado. Processando diretórios...")
        true_labels, predicted_labels = process_directories(sys.argv[1], sys.argv[2], model)
        cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])
        print_and_save_confusion_matrix(cm, true_labels, predicted_labels, 'confusion_matrix.txt')
        print("Matriz de confusão e métricas salvas em 'confusion_matrix.txt'.")
