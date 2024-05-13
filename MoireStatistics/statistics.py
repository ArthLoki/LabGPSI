from moire.moire_detection_statistics import load_and_evaluate
import os
from time import sleep
import argparse

REPORT_FOLDER = 'report'

## Create folders if they don't exist
if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)


# Statistics File Generators
def generateStatisticsFile(counter_cm, counter_channels_cm, channels_metrics, falses_cm, img_folder, positive):
    print(f'\n\nGenerating {"Positive" if positive else "Negative"} Statistics...')
    filename = f'statistics_{"positive" if positive else "negative"}.txt'

    file = open(os.path.join(REPORT_FOLDER, filename), 'w')

    # Confusion Matrix
    file.write(f'1. Confusion Matrix Data from all {len(os.listdir(img_folder))} {"positive" if positive else "negative"} images\n\n')
    for metric, value in counter_cm.items():
        if metric in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score']:
            file.write(f'{metric}: {value/len(os.listdir(img_folder))}\n')
        else:
            file.write(f'{metric}: {value}\n')

    # Confusion Matrix per channel
    file.write(f'\n\n2. Confusion Matrix Data and Metrics from all channels of all {len(os.listdir(img_folder))} {"positive" if positive else "negative"} images\n')
    for channel, dict_cm in counter_channels_cm.items():
        file.write(f'\n-----> Channel {channel}:\n\n')
        for cm, value in dict_cm.items():
            file.write(f'{cm}: {value}\n')
        for metric, value in channels_metrics[channel].items():
            file.write(f'{metric}: {value}\n')

    # False Negatives/Positives Images
    file.write(f'\n\n3. List of false {"negative" if positive else "positive"} images:\n\n')
    for image_name, channels in falses_cm.items():
        file.write(f'{image_name}: {channels}\n')

    file.write(f'\nTotal False {"Negative" if positive else "Positive"}: {len(falses_cm.keys())}')
    file.write(f'\nTotal True {"Positive" if positive else "Negative"}: {len(os.listdir(img_folder))-len(falses_cm.keys())}\n')

    # Tabela Canal x Confusion Matrix
    file.write(f'\n\n4. Channel x Confusion Matrix Table:\n\n')

    channels_labels = list(counter_channels_cm.keys())  # row's label
    metrics_labels = [metric.split(' ')[0][0]+metric.split(' ')[1][0]
                    for metric in list(counter_channels_cm[channels_labels[0]].keys())] # column's label

    file.write(' '*((2*len(channels_labels[0])) + 1))
    file.write(f'  '.join(metrics_labels))

    for channel in channels_labels:
        metrics = '  '.join([str(num) for num in list(counter_channels_cm[channel].values())])
        file.write(f"\n{channel} | {metrics}")

    file.close()
    return

def generateStatisticsFileV2(counter_channels_cm, falses_cm, img_folder, positive):

    print(f'\n\nGenerating {"Positive" if positive else "Negative"} Statistics V2...')
    filename = f'statistics_{"positive" if positive else "negative"}_v2.txt'

    file = open(os.path.join(REPORT_FOLDER, filename), 'w')

    file.write('TOTAIS\n')
    file.write(f'Diretorio {"Positivas" if positive else "Negativas"} - {len(os.listdir(img_folder))} (total de fotos no diretorio de {"positivas" if positive else "negativas"})\n')
    file.write(f'TP - {len(os.listdir(img_folder))-len(falses_cm.keys())} (total de fotos do diretorio consideradas {"positivas" if positive else "negativas"})\n')
    file.write(f'FN - {len(falses_cm.keys())} (total de falso {"negativas" if positive else "positivas"})\n')
    file.write(f'Lista de fotos falso {"negativas" if positive else "positivas"} -\n')
    for image_name, channels in falses_cm.items():
        file.write(f'{image_name}\n')

    file.write(f'\nCANAIS\n')
    '''
    Estrutura de counter_channels_counter:

        counter_channels_cm = {
            'LL': {"True Positive": <counter>, 'True Negative': <counter>, "False Positive": <counter>, "False Negative": <counter>},
            'LH': {"True Positive": <counter>, 'True Negative': <counter>, "False Positive": <counter>, "False Negative": <counter>},
            'HL': {"True Positive": <counter>, 'True Negative': <counter>, "False Positive": <counter>, "False Negative": <counter>},
            'HH': {"True Positive": <counter>, 'True Negative': <counter>, "False Positive": <counter>, "False Negative": <counter>},
        }
    '''

    channels_labels = list(counter_channels_cm.keys())
    for channel in channels_labels:
        metrics_values = list(counter_channels_cm[channel].values())
        if positive:
            metrics = ['TP', 'FN']
            values = [metrics_values[0], metrics_values[3]]
        else:
            metrics = ['TN', 'FP']
            values = [metrics_values[1], metrics_values[2]]

        file.write(f"{channel} -\t{metrics[0]} - {values[0]}\t{metrics[1]} - {values[1]}\n")

    file.close()
    return


# Calculating Data to Save
def get_falses_cm(filename, channel, result, positive, falses_cm):
    # Save filenames of the false negative or the false positive images
    if positive:
        if result[False] > 0:
            if filename not in falses_cm.keys():
                falses_cm[filename] = [channel]
            else:
                falses_cm[filename].append(channel)
    else:
        if result[True] > 0:
            if filename not in false_positives:
                falses_cm[filename] = [channel]
            else:
                falses_cm[filename].append(channel)
    return falses_cm

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


# Main Statistics
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
            status = load_and_evaluate(model_path, img_folder, file)

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

        # generateStatisticsFile(counter_cm, counter_channels_cm, channels_metrics, falses_cm, img_folder, positive)
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
    parser.add_argument("model_path", type=str, help="Caminho para o arquivo do modelo treinado (.h5).")
    args = parser.parse_args()

    statistics(args.img_folder, args.model_path)
