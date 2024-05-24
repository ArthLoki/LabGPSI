# from moire.moire_detection_statistics import load_and_evaluate
from moire.new_test_moire import load_and_evaluate, load_trained_model
from statisticsFileGeneration import (generateStatisticsFileV3,
                                      # generateStatisticsFileV2, generateStatisticsFile,
                                      # get_falses_cm, update_counter_channels_cm, calculate_metrics
                                      )
import os
from time import sleep
import argparse
# from keras.models import load_model


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

        results_images = []

        index_current_file = 0
        positive = False  # inicializa a variável como False
        for file in image_filenames:
            if 'positive' in list_img_folder[-1].lower():
                positive = True
            elif 'negative' in list_img_folder[-1].lower():
                positive = False
            else:
                print("The image must have a previous classification.\n")
                exit(1)

            index_current_file += 1
            print(f"\n\nProcessing {'positive' if positive else 'negative'} file ({file}): {index_current_file}/{len(os.listdir(img_folder))}...")

            # Chamada Moire
            status = load_and_evaluate(model, img_folder, file)
            print('status: ', status)

            # Check if status returned an error
            if status.get('Error') is not None:
                print(status.get('Error'))
                exit(1)

            # Get results from channels and confusion matrix to create statistics
            result = status.get('moire')
            results_images.append(result)

            print(f"Finished processing {'positive' if positive else 'negative'} file {index_current_file}/{len(os.listdir(img_folder))}!")
            sleep(0.5)

        print(results_images)
        generateStatisticsFileV3(image_filenames, results_images, positive)

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
