from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from moire.moire_detection import load_and_evaluate

import os
from datetime import datetime

UPLOAD_FOLDER = 'uploads/images'

UPLOAD_FOLDER_TEST = str(os.path.join(UPLOAD_FOLDER, 'test')).replace('\\', '/')
UPLOAD_FOLDER_TEST_PROCESSED = str(os.path.join(UPLOAD_FOLDER, 'test_processed')).replace('\\', '/')

MODEL_FOLDER = 'uploads/savedmodels'

REPORT_FOLDER = 'report'
REPORT_FOLDER_BY_IMAGE = str(os.path.join(REPORT_FOLDER, 'by_image')).replace('\\', '/')
REPORT_FOLDER_BY_CHANNEL = str(os.path.join(REPORT_FOLDER, 'by_channel')).replace('\\', '/')
REPORT_FOLDER_CM = str(os.path.join(REPORT_FOLDER, 'confusion_matrix')).replace('\\', '/')

ALLOWED_EXTENSIONS_FILE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_MODEL = {'h5', 'keras'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER_TEST'] = UPLOAD_FOLDER_TEST
app.config['UPLOAD_FOLDER_TEST_PROCESSED'] = UPLOAD_FOLDER_TEST_PROCESSED

app.config['MODEL_FOLDER'] = MODEL_FOLDER

app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['REPORT_FOLDER_BY_IMAGE'] = REPORT_FOLDER_BY_IMAGE
app.config['REPORT_FOLDER_BY_CHANNEL'] = REPORT_FOLDER_BY_CHANNEL
app.config['REPORT_FOLDER_CM'] = REPORT_FOLDER_CM

app.config['SECRET_KEY'] = 'super-secret'
app.config['FLASK_ENV'] = 'development'  # Set Flask environment to development
app.debug = app.config['FLASK_ENV'] == 'development' # Enable Flask debug mode

## Create folders if they don't exist
if not os.path.exists(app.config['MODEL_FOLDER']):
    os.makedirs(app.config['MODEL_FOLDER'])

if not os.path.exists(app.config['UPLOAD_FOLDER_TEST']):
    os.makedirs(app.config['UPLOAD_FOLDER_TEST'])

if not os.path.exists(app.config['UPLOAD_FOLDER_TEST_PROCESSED']):
    os.makedirs(app.config['UPLOAD_FOLDER_TEST_PROCESSED'])

if not os.path.exists(app.config['REPORT_FOLDER']):
    os.makedirs(app.config['REPORT_FOLDER'])

if not os.path.exists(app.config['REPORT_FOLDER_BY_IMAGE']):
    os.makedirs(app.config['REPORT_FOLDER_BY_IMAGE'])

if not os.path.exists(app.config['REPORT_FOLDER_BY_CHANNEL']):
    os.makedirs(app.config['REPORT_FOLDER_BY_CHANNEL'])

if not os.path.exists(app.config['REPORT_FOLDER_CM']):
    os.makedirs(app.config['REPORT_FOLDER_CM'])


## Check if the file extension is allowed
def allowed_file(filename, filetype):
    allowed_extensions = ALLOWED_EXTENSIONS_FILE if filetype == 0 else ALLOWED_EXTENSIONS_MODEL

    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        # CHECK MODEL FILE
        # If the model folder is empty, the user needs to upload a model. If there's a model in the folder,
        # the upload button is disabled and there's no need to upload it again
        # If the model upload is needed, all the uploads (model + image) will take more time to finish
        if len(os.listdir(app.config['MODEL_FOLDER'])) == 0:
            # check if the post request has the file part
            saved_model = request.files['savedmodel']
            if saved_model.filename == '':
                return jsonify({'Error': 'No selected model'})

            if saved_model and allowed_file(saved_model.filename, 1):
                clear_model_folder()
                model_name = secure_filename(str(saved_model.filename))
                if not os.path.exists(os.path.join(app.config['MODEL_FOLDER'], model_name)):
                    saved_model.save(os.path.join(app.config['MODEL_FOLDER'], model_name))
            else:
                return jsonify({'Error': 'Model file not allowed'})
        else:
            model_name = os.listdir(app.config['MODEL_FOLDER'])[0] if os.listdir(app.config['MODEL_FOLDER']) else None

        # CHECK IMAGE FILE
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'Error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'Error': 'No selected file'})

        if file and allowed_file(file.filename, 0) and model_name is not None:
            filename = secure_filename(str(file.filename))

            file.save(os.path.join(app.config['UPLOAD_FOLDER_TEST'], filename))

            # CHAMADA MOIRE
            model_path = str(os.path.join(app.config['MODEL_FOLDER'], model_name)).replace('\\', '/')

            if len(os.listdir(app.config['UPLOAD_FOLDER_TEST'])) > 0:
                status = load_and_evaluate(model_path,
                                                app.config['UPLOAD_FOLDER_TEST'])

                # Check if status returned an error
                if status.get('Error') is not None:
                    clear_test_folders()
                    return jsonify(status)

                # Get results from predictions to create statistics
                results_predictions = status.get('results_predictions')

                # Returns a json error if there's no 'results_predictions' in status
                if results_predictions is None:
                    clear_test_folders()
                    return jsonify({'Error': 'results_predictions is None'})

                # Get results from confusion matrix to create statistics
                results_cm = status.get('results_cm')

                # Returns a json error if there's no 'results_cm' in status
                if results_cm is None:
                    clear_test_folders()
                    return jsonify({'Error': 'results_cm is None'})

                # generateReportByImage(filename, results_predictions)
                # generateReportByChannel(filename, results_predictions)
                generateReportCM(filename, results_cm)

                # Clear folders after use images
                clear_test_folders()

                return jsonify ({
                                'filename': filename,
                                'moire': status.get('moire'),
                                'model_name': model_name,
                                'datetime': datetime.now(),
                                })
            return jsonify({'Error': 'No file found'})

    return render_template('index.html', empty_model_folder=len(os.listdir(app.config['MODEL_FOLDER'])) == 0)


def clear_test_folders():
    if len(os.listdir(app.config['UPLOAD_FOLDER_TEST'])) > 0:
        for filename_image in os.listdir(app.config['UPLOAD_FOLDER_TEST']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER_TEST'], filename_image))

    if len(os.listdir(app.config['UPLOAD_FOLDER_TEST_PROCESSED'])) > 0:
        for filename_tiff in os.listdir(app.config['UPLOAD_FOLDER_TEST_PROCESSED']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER_TEST_PROCESSED'], filename_tiff))
    return

def clear_model_folder():
    if len(os.listdir(app.config['MODEL_FOLDER'])) > 0:
        for model_name in os.listdir(app.config['MODEL_FOLDER']):
            os.remove(os.path.join(app.config['MODEL_FOLDER'], model_name))
    return

def generateReportByImage(image_name, results_predictions):
    """
    Structure of the report_<image_name>.txt file

    <image_name>:

    channel 1 (<channel_name>): status.get('results_predictions')[0]
    .
    .
    .
    channel 12 (<channel_name>): status.get('results_predictions')[11]
    """
    filename = 'report_'+image_name.split('.')[0]+'.txt'
    file = open(os.path.join(app.config['REPORT_FOLDER_BY_IMAGE'], filename), 'w')

    if filename in os.listdir(app.config['REPORT_FOLDER_BY_IMAGE']):
        # file.write(f'{upper(image_name)}:\n')
        for i, channel in enumerate(os.listdir(app.config['UPLOAD_FOLDER_TEST_PROCESSED'])):
            file.write(f'Channel {i+1} ({channel}): {results_predictions[i]}\n')
    return


def generateReportByChannel(image_name, results_predictions):
    """
    Structure of the report_channel_<i+1>.txt file

    Channel <i+1>:
    <image_name_1> (<channel_name_1>): status.get('results_predictions')[0]
    <image_name_2> (<channel_name_2>): status.get('results_predictions')[1]
    .
    .
    .
    <image_name_<i+1>> (<channel_name_<i+1>>): status.get('results_predictions')[i]
    """

    channels_name = os.listdir(app.config['UPLOAD_FOLDER_TEST_PROCESSED'])
    for i, channel in enumerate(channels_name):
        filename = f'report_channel_{i+1}.txt'
        if filename in os.listdir(app.config['REPORT_FOLDER_BY_CHANNEL']):
            print(f"Updating channel {i + 1} report file.\nProgress: {i + 1} of {len(channels_name)}")
            file = open(os.path.join(app.config['REPORT_FOLDER_BY_CHANNEL'], filename), 'a')

            # Write images predictions data
            file.write(f'{image_name} ({channels_name[i]}): {results_predictions[i]}\n')
            print(f"Channel {i + 1} report file updated.\n")
        else:
            print(f"Creating channel {i+1} report file.\nProgress: {i+1} of {len(channels_name)}")
            file = open(os.path.join(app.config['REPORT_FOLDER_BY_CHANNEL'], filename), 'w')

            # Write images predictions data
            file.write(f'{image_name} ({channels_name[i]}): {results_predictions[i]}\n')
            print(f"Channel {i + 1} report file created.\n")
    return


def generateReportCM(image_name, results_cm):
    """
    Structure of the report_<image_name>_cm.txt file

    Confusion Matrix Data:
        True Positives: <TP>
        True Negatives: <TN>
        False Positives: <FP>
        False Negatives: <FN>
        Accuracy: <accuracy>
        Precision: <precision>
        Recall: <recall>
        F1 Score: <f1score>
    """
    filename = 'report_' + image_name.split('.')[0] + '_cm.txt'
    file = open(os.path.join(app.config['REPORT_FOLDER_CM'], filename), 'w')

    if filename in os.listdir(app.config['REPORT_FOLDER_CM']):
        # file.write('\nConfusion Matrix Data:\n')
        for _, content in enumerate(results_cm.items()):
            file.write(f'{content[0]}: {content[1]}\n')
    return


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
