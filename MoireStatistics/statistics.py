# from moire.moire_detection_statistics import load_and_evaluate
from moire.new_test_moire import load_and_evaluate
from statisticsFileGeneration import (generateStatisticsFileV2,
                                      generateStatisticsFile,
                                      get_falses_cm,
                                      update_counter_channels_cm,
                                      calculate_metrics)
import os
from time import sleep
import argparse
from moire.mCNN import createModel  # Importando a função de criação do modelo
from keras.models import load_model


def load_trained_model(weights_path):
    try:
        model = createModel(375, 500, 1, 1)  # Assumindo que estas são as dimensões esperadas pelo modelo
        model.load_weights(weights_path)
        return model
    except Exception as e:
        return {'Error': 'load_trained_model: ' + str(e)}


# Main Statistics
def statistics(img_folder, model_path):
    # Load model
    model = load_trained_model(model_path)

    if isinstance(model, dict) and 'Error' in model:
        print(model['Error'])
        return -1
    else:
        print('Model loaded successfully.')

    # Images files
    image_filenames = os.listdir(img_folder)

    if len(image_filenames) > 0:
        if '\\' in img_folder:
            list_img_folder = img_folder.split('\\')
        else:
            list_img_folder = img_folder.split('/')

        counter_channels_boolean = {
            "LL": {True: 0, False: 0},
            "LH": {True: 0, False: 0},
            "HL": {True: 0, False: 0},
            "HH": {True: 0, False: 0},
        }

        counter_cm = {
            "True Positive": 0,
            "True Negative": 0,
            "False Positive": 0,
            "False Negative": 0,
            "Accuracy": 0,
            'Precision': 0,
            'Recall (Sensitivity)': 0,
            'F1 Score': 0,
        }

        falses_cm = {}

        index_current_file = 0
        positive = False  # inicializa a variável como False
        for file in image_filenames:
            if list_img_folder[-1].lower() == 'positive':
                positive = True
            elif list_img_folder[-1].lower() == 'negative':
                positive = False
            else:
                print("The image must have a previous classification.\n")
                exit(1)

            index_current_file += 1
            print(f"\n\nProcessing {'positive' if positive else 'negative'} file ({file}): {index_current_file}/{len(os.listdir(img_folder))}...")

            # Chamada Moire
            status = load_and_evaluate(model, img_folder, file)

            # Check if status returned an error
            if status.get('Error') is not None:
                print(status.get('Error'))
                exit(1)

            # Get results from channels and confusion matrix to create statistics
            results_channels = status.get('results_channels')
            results_cm = status.get('results_cm')

            # Checks if none of the results is None and increment counter
            if results_channels is not None:
                for channel, result in results_channels.items():
                    falses_cm = get_falses_cm(file, channel, result, positive, falses_cm)
                    for key, count in result.items():
                        counter_channels_boolean[channel][key] += count
            else:
                print("ERROR in statistics: results_channels cannot be None.\n")
                exit(1)

            if results_cm is not None:
                for key, content in results_cm.items():
                    counter_cm[key] += content
            else:
                print("ERROR in statistics: results_cm cannot be None.\n")
                exit(1)

            print(f"Finished processing {'positive' if positive else 'negative'} file {index_current_file}/{len(os.listdir(img_folder))}!")
            sleep(0.5)

        counter_channels_cm = update_counter_channels_cm(counter_channels_boolean, positive)
        channels_metrics = calculate_metrics(counter_channels_cm)

        generateStatisticsFile(counter_cm, counter_channels_cm, channels_metrics, falses_cm, img_folder, positive)
        generateStatisticsFileV2(counter_channels_cm, falses_cm, img_folder, positive)

        # If you wanto to join both positive and negative files, run in terminal
        # 'python ./statisticsFile.py <filename_positive> <filename_negative> <filename_both>'
        # Sometimes, it's python3 instead of python

        # OBS: All the filenames must be a path to the .txt fileS
    else:
        print("Given folder is empty.\n")
        exit(1)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detecta padrões de moiré em imagens usando a transformada de Haar e uma rede neural.")
    parser.add_argument("img_folder", type=str, help="Caminho para a pasta das imagens a serem analisadas.")
    parser.add_argument("model_path", type=str, help="Caminho para o arquivo do modelo treinado (.keras).")
    args = parser.parse_args()

    statistics(args.img_folder, args.model_path)
