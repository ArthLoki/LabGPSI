from moire.moire_detection_statistics import load_and_evaluate
import os

TEST_FOLDER = 'images/test'
TEST_FOLDER_POSITIVE = 'images/test/positive'
TEST_FOLDER_NEGATIVE = 'images/test/negative'
MODEL_FOLDER = 'savedmodels'
REPORT_FOLDER = 'report'

## Create folders if they don't exist
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

if not os.path.exists(TEST_FOLDER):
    os.makedirs(TEST_FOLDER)
    os.makedirs(TEST_FOLDER_NEGATIVE)
    os.makedirs(TEST_FOLDER_POSITIVE)

if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

def statistics(positive=True):
    # Model file and Images files
    image_filenames = os.listdir(TEST_FOLDER_POSITIVE) if positive else os.listdir(TEST_FOLDER_NEGATIVE)
    img_folder = TEST_FOLDER_POSITIVE if positive else TEST_FOLDER_NEGATIVE
    model_name = os.listdir(MODEL_FOLDER)[0] if os.listdir(MODEL_FOLDER) else None

    counter_channels_boolean = {
        "LL": {True: 0, False: 0},
        "LH": {True: 0, False: 0},
        "HL": {True: 0, False: 0},
        "HH": {True: 0, False: 0},
    }

    counter_cm = {
        "True Positives": 0,
        "True Negatives": 0,
        "False Positives": 0,
        "False Negatives": 0,
        "Accuracy": 0,
        'Precision': 0,
        'Recall (Sensitivity)': 0,
        'F1 Score': 0,
    }

    for file in image_filenames:
        if model_name is not None:
            # Chamada Moire
            model_path = str(os.path.join(MODEL_FOLDER, model_name)).replace('\\', '/')
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

            print('counter_channels_boolean:', counter_channels_boolean)
            print('counter cm:', counter_cm)

    counter_channels_cm = update_counter_channels_cm(counter_channels_boolean, positive)
    channels_metrics = calculate_metrics(counter_channels_cm, img_folder)
    return

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
                    counter_channels_boolean[channel]["True Positive" if positive else "False Positive"] += count
                else:
                    counter_channels_boolean[channel]["False Negative" if positive else "True Negative"] += count

    return counter_channels_cm

def calculate_metrics(counter_channels_cm, img_folder):
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

            accuracy = (TP + TN) / (TP + TN + FP + FN) \
                if TP + TN + FP + FN > 0 else (TP + TN) / len(os.listdir(img_folder))

            precision = TP / (TP + FP) if TP + FP > 0 else TP

            recall = TP / (TP + FN) if TP + FN > 0 else TP

            f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 2 * (
                    precision * recall)

            channels_metrics[channel]["Accuracy"] = accuracy
            channels_metrics[channel]["Precision"] = precision
            channels_metrics[channel]["Recall (Sensitivity)"] = recall
            channels_metrics[channel]["F1 Score"] = f1score

    return channels_metrics