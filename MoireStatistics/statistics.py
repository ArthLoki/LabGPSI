from moire.moire_detection_statistics import load_and_evaluate
import os

TEST_FOLDER = 'images/test'
TEST_FOLDER_POSITIVE = 'images/test/positive'
TEST_FOLDER_NEGATIVE = 'images/test/negative'

MODEL_FOLDER = 'savedmodels'
REPORT_FOLDER = 'report'

ALLOWED_EXTENSIONS_FILE = {'png', 'jpg', 'jpeg', 'JPG'}
ALLOWED_EXTENSIONS_MODEL = {'h5', 'keras'}

## Create folders if they don't exist
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

if not os.path.exists(TEST_FOLDER):
    os.makedirs(TEST_FOLDER)
    os.makedirs(TEST_FOLDER_NEGATIVE)
    os.makedirs(TEST_FOLDER_POSITIVE)

if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)


## Check if the file extension is allowed
def allowed_file(filename, filetype):
    allowed_extensions = ALLOWED_EXTENSIONS_FILE if filetype == 0 else ALLOWED_EXTENSIONS_MODEL

    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generateStatisticsFile(results_cm, results_channels, positive=True):
    """
        Structure of the statistics.txt file

        "LL True": <counter>
        "LL False": <counter>
        "LH True": <counter>
        "LH False": <counter>
        "HL True": <counter>
        "HL False": <counter>
        "HH True": <counter>
        "HH False": <counter>
        True Positives: <TP>
        True Negatives: <TN>
        False Positives: <FP>
        False Negatives: <FN>
        Accuracy: <accuracy>/<total_files>
        Precision: <precision>/<total_files>
        Recall (Sensitivity): <recall>/<total_files>
        F1 Score: <f1score>/<total_files>

        Total Files: <total_files>
        """

    filename = 'statistics_positive.txt' if positive else 'statistics_negative.txt'

    # Checking if the file exists
    if filename not in os.listdir(REPORT_FOLDER):
        createStatisticsFile(filename, results_cm, results_channels)
    else:
        updateStatisticsFile(filename, results_cm, results_channels)
    return

def createStatisticsFile(filename, results_cm, results_channels):
    # Creating file, because it does not exist
    file = open(REPORT_FOLDER + filename, 'w')
    total_files = 1

    for channel, result in results_channels.items():
        file.write(f'{channel}: {result}\n')

    for key, content in results_cm.items():
        if key in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score']:
            file.write(f'{key}: {content / total_files}\n')
        else:
            file.write(f'{key}: {content}\n')

    file.write(f'Total Files: {total_files}\n')

    file.close()
    return

def updateStatisticsFile(filename, results_cm, results_channels):
    current_data = getCurrentStatisticsFileData(filename)
    total_files = int(current_data.get("Total Files")) + 1 if current_data.get("Total Files") is not None else 1

    # Opens the file to update the current data
    file = open(REPORT_FOLDER + filename, 'w')

    for channel, result in results_channels.items():
        current_channel_data = current_data.get(channel)
        if current_channel_data is not None:
            file.write(f'{channel}: {int(current_channel_data) + result}\n')

    for key, content in results_cm.items():
        current_cm_data = current_data.get(key)
        if current_cm_data is not None:
            if key in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score']:
                file.write(f'{key}: {(int(current_cm_data) / total_files) + content}\n')
            else:
                file.write(f'{key}: {int(current_cm_data) + content}\n')

    file.write(f'Total Files: {total_files}\n')

    file.close()
    return

def getCurrentStatisticsFileData(filename):
    # Opens the file to read the current data
    file = open(REPORT_FOLDER + filename, 'r')

    # Variable to save the data
    current_data = {}

    # Gets file data of each line
    for line in file:
        print(line)
        lDados = line.strip().split(": ")
        current_data[lDados[0]] = lDados[1]

    file.close()
    return current_data

def statistics(positive=True):
    # Model file and Images files
    model_name = os.listdir(MODEL_FOLDER)[0] if os.listdir(MODEL_FOLDER) else None
    files = os.listdir(TEST_FOLDER_POSITIVE) if positive else os.listdir(TEST_FOLDER_NEGATIVE)

    for file in files:
        if allowed_file(file, 0) and model_name is not None:
            # Chamada Moire
            model_path = str(os.path.join(MODEL_FOLDER, model_name)).replace('\\', '/')
            status = load_and_evaluate(model_path, TEST_FOLDER, file)

            # Check if status returned an error
            if status.get('Error') is not None:
                print(status.get('Error'))
                break

            # Get results from channels and confusion matrix to create statistics
            results_channels = status.get('results_channels')
            results_cm = status.get('results_cm')

            # Checks if none of the results is None
            if results_channels is not None and results_cm is not None:
                generateStatisticsFile(results_cm, results_channels, positive)
            else:
                print('results_channels or results_cm is None')
                break
        else:
            print('Extension file not allowed or model_name is None')
            break
    return

def main():
    # Testing for positive images
    statistics()

    # Testing for negative images
    statistics(positive=False)
    return

if __name__ == '__main__':
    main()
