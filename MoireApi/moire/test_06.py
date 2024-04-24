import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import logging
import os

WIDTH = 750
HEIGHT = 1000
TRANSFORM_COUNT = 12  # Número de transformações por imagem original

def load_and_evaluate(model_path, test_dir):
    logging.basicConfig(level=logging.INFO)

    # Carrega o modelo
    try:
        logging.info("Carregando o modelo...")
        model = load_model(model_path)
        logging.info("Modelo carregado com sucesso.")
    except Exception as e:
        return {'Error': f'Error in line 19 (loading {model_path.split("/")[-1]} in test_06.py): {e}'}

    # Configura o gerador de dados de teste para imagens em escala de cinza
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(HEIGHT, WIDTH),
        batch_size=1,  # 32
        class_mode=None,  # 'binary'
        color_mode='grayscale',
        shuffle=False
    )

    # Calcula o número de passos por época
    steps_per_epoch = np.ceil(test_generator.samples / test_generator.batch_size)

    if steps_per_epoch == 0:
        steps_per_epoch = 10
        # return 'Error in line 36 (getting number of steps per epoch in test_06.py): steps_per_epoch is zero'

    # Avalia o modelo
    try:
        logging.info("Avaliando o modelo...")
        model.evaluate(test_generator, steps=steps_per_epoch)
    except Exception as e:
        return {'Error': f'Error in line 45 (evaluating {model_path.split("/")[-1]} in test_06.py): {e}'}

    # Previsões do conjunto de teste
    try:
        logging.info("Gerando previsões para o conjunto de teste...")
        test_generator.reset()
        predictions = model.predict(test_generator, steps=steps_per_epoch)
    except Exception as e:
        return {'Error': f'Error in line 53 (generating predictions for {model_path.split("/")[-1]} in test_06.py): {e}'}

    # Generating array of predictions
    predictions = np.round(predictions).astype(int).flatten()

    # Processar as previsões em blocos de 12
    grouped_predictions = {}
    for i in range(0, len(test_generator.filenames), TRANSFORM_COUNT):
        # Assegurando que cada bloco de 12 previsões pertença à mesma imagem original
        base_name = test_generator.filenames[i].rsplit('_', 1)[0]  # Remove a última parte após o underscore
        grouped_predictions[base_name] = predictions[i:i + TRANSFORM_COUNT]

    # Decidir a classificação final com base nos blocos de 12
    final_predictions = []
    final_true_classes = []
    for i in range(0, len(test_generator.filenames), TRANSFORM_COUNT):
        group = grouped_predictions[test_generator.filenames[i].rsplit('_', 1)[0]]
        # Se qualquer uma das transformações for positiva, classificar como positivo
        final_pred = 1 if 1 in group else 0
        final_predictions.append(final_pred)
        
        # Como as classes verdadeiras são as mesmas para todas as 12 transformações,
        # podemos simplesmente pegar a classe do primeiro arquivo do bloco.
        true_class = test_generator.classes[i]
        final_true_classes.append(true_class)

    # Calculando a matriz de confusão e o relatório de classificação
    logging.info("Calculando a matriz de confusão e o relatório de classificação...")
    cm = confusion_matrix(final_true_classes, final_predictions)
    class_report = classification_report(final_true_classes, final_predictions, target_names=['Negative', 'Positive'])

    # Salvando os resultados em um arquivo
    with open('model_evaluation.txt', 'w') as f:
        f.write(f"Model tested: {model_path}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"[[True Negative, False Positive]\n [False Negative, True Positive]]\n")
        f.write(f"Classification Report:\n{class_report}\n")

    logging.info("Resultados salvos e processo concluído.")

    return {
        # "loss": results[0],
        # "accuracy": results[1],
        # 'confusion_matrix': cm,
        'classification_report': class_report
    }

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Test a neural network model on a new set of images.')
#     parser.add_argument('model_path', type=str, help='Path to the complete .h5 model file.')
#     parser.add_argument('test_dir', type=str, help='Directory with test data.')
#     args = parser.parse_args()

#   load_and_evaluate(args.model_path, args.test_dir)
