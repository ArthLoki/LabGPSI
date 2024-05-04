from moire.moire_detection_statistics import load_and_evaluate
import os
from time import sleep
import argparse

REPORT_FOLDER = 'report'

## Create folders if they don't exist
if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

def update_counter_channels_cm(counter_channels_boolean, positive):

    counter_channels_cm = {
        'LL': {"True Positive": 0, 'True Negative': 0, "False Positive": 0, "False Negative": 0},
        'LH': {"True Positive": 0, 'True Negative': 0, "False Positive": 0, "False Negative": 0},
        'HL': {"True Positive": 0, 'True Negative': 0, "False Positive": 0, "False Negative": 0},
        'HH': {"True Positive": 0, 'True Negative': 0, "False Positive": 0, "False Negative": 0},
    }

    for channel, dict_boolean in counter_channels_boolean.items():
        dict_channel_cm = counter_channels_cm.get(channel)
        if dict_channel_cm is not None:
            for boolean, count in dict_boolean.items():
                if boolean:
                    counter_channels_cm[channel]["True Positive" if positive else "False Positive"] += count
                else:
                    counter_channels_cm[channel]["False Negative" if positive else "True Negative"] += count

    return counter_channels_cm

def calculate_metrics(counter_channels_cm):
    channels_metrics = {
        "LL": {"Accuracy": 0, 'Precision': 0, 'Recall (Sensitivity)': 0, 'F1 Score': 0},
        "LH": {"Accuracy": 0, 'Precision': 0, 'Recall (Sensitivity)': 0, 'F1 Score': 0},
        "HL": {"Accuracy": 0, 'Precision': 0, 'Recall (Sensitivity)': 0, 'F1 Score': 0},
        "HH": {"Accuracy": 0, 'Precision': 0, 'Recall (Sensitivity)': 0, 'F1 Score': 0},
    }

    for channel, dict_channel_cm in counter_channels_cm.items():
        dict_channels_metrics = channels_metrics.get(channel)
        if dict_channels_metrics is not None:
            TN = dict_channel_cm.get("True Negative")
            FP = dict_channel_cm.get("False Positive")
            FN = dict_channel_cm.get("False Negative")
            TP = dict_channel_cm.get("True Positive")

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if (TP+FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP+FN) > 0 else 0.0
            f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            channels_metrics[channel]["Accuracy"] = accuracy
            channels_metrics[channel]["Precision"] = precision
            channels_metrics[channel]["Recall (Sensitivity)"] = recall
            channels_metrics[channel]["F1 Score"] = f1score

    return channels_metrics

def generateStatisticsFile(counter_cm, counter_channels_cm, channels_metrics, img_folder, positive):

    print(f'\n\nGenerating {"Positive" if positive else "Negative"} Statistics...')
    filename = f'statistics_{"positive" if positive else "negative"}.txt'

    file = open(os.path.join(REPORT_FOLDER, filename), 'w')

    file.write(f'1. Confusion Matrix Data from all {len(os.listdir(img_folder))} {"positive" if positive else "negative"} images\n\n')
    for metric, value in counter_cm.items():
        if metric in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score']:
            file.write(f'{metric}: {value/len(os.listdir(img_folder))}\n')
        else:
            file.write(f'{metric}: {value}\n')

    file.write(f'\n\n2. Confusion Matrix Data and Metrics from all channels of all {len(os.listdir(img_folder))} {"positive" if positive else "negative"} images\n')
    for channel, dict_cm in counter_channels_cm.items():
        file.write(f'\n-----> Channel {channel}:\n\n')
        for cm, value in dict_cm.items():
            file.write(f'{cm}: {value}\n')
        for metric, value in channels_metrics[channel].items():
            file.write(f'{metric}: {value}\n')
    file.close()
    return

def statistics(img_folder, model_path):
    # Model file and Images files
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

        index_current_file = 0
        positive = False  # inicializa a variável como False
        for file in image_filenames:
            if list_img_folder[-1].lower() == 'positive':
                positive = True
            elif list_img_folder[-1].lower() == 'negative':
                positive = False
            else:
                print("The image must have a previous classification.\n")
                break

            index_current_file += 1
            print(f"\n\nProcessing {'positive' if positive else 'negative'} file ({file}): {index_current_file}/{len(os.listdir(img_folder))}...")

            # Chamada Moire
            status = load_and_evaluate(model_path, img_folder, file)

            # Check if status returned an error
            if status.get('Error') is not None:
                print(status.get('Error'))
                break

            # Get results from channels and confusion matrix to create statistics
            results_channels = status.get('results_channels')
            results_cm = status.get('results_cm')

            # Checks if none of the results is None and increment counter
            if results_channels is not None:
                for channel, result in results_channels.items():
                    for key, count in result.items():
                        counter_channels_boolean[channel][key] += count

            if results_cm is not None:
                for key, content in results_cm.items():
                    counter_cm[key] += content

            print(f"\nFinished processing {'positive' if positive else 'negative'} file {index_current_file}/{len(os.listdir(img_folder))}!")
            sleep(0.5)

        counter_channels_cm = update_counter_channels_cm(counter_channels_boolean, positive)
        channels_metrics = calculate_metrics(counter_channels_cm)
        generateStatisticsFile(counter_cm, counter_channels_cm, channels_metrics, img_folder, positive)
    else:
        print("Given folder is empty.\n")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detecta padrões de moiré em imagens usando a transformada de Haar e uma rede neural.")
    parser.add_argument("img_folder", type=str, help="Caminho para a pasta das imagens a serem analisadas.")
    # parser.add_argument("positive", type=bool, help="Indica se possui ou nao padrao de Moire.")
    parser.add_argument("model_path", type=str, help="Caminho para o arquivo do modelo treinado (.h5).")
    args = parser.parse_args()

    statistics(args.img_folder, args.model_path)