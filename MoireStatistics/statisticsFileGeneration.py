import os

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


def generateStatisticsFileV3(images_filenames, results_images, positive):
    print(f'\n\nGenerating {"Positive" if positive else "Negative"} Statistics V3...')
    filename = f'statistics_{"positive" if positive else "negative"}_v3.txt'

    file = open(os.path.join(REPORT_FOLDER, filename), 'w')

    for i in range(len(images_filenames)):
        if positive:
            if not results_images[i]: cm_final_result = 'FN'
            else: cm_final_result = 'TP'
        else:
            if not results_images[i]: cm_final_result = 'TN'
            else: cm_final_result = 'FP'

        content = f'Image {images_filenames[i]} ({i+1}/{len(images_filenames)}): {results_images[i]} ({cm_final_result})\n'
        file.write(content)

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
