from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from moire.moire_detection_api import load_and_evaluate

import os
from datetime import datetime

UPLOAD_FOLDER_TEST = 'uploads/images/test'
MODEL_FOLDER = 'uploads/savedmodels'

ALLOWED_EXTENSIONS_FILE = {'png', 'jpg', 'jpeg', 'JPG'}
ALLOWED_EXTENSIONS_MODEL = {'h5', 'keras'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER_TEST'] = UPLOAD_FOLDER_TEST
app.config['MODEL_FOLDER'] = MODEL_FOLDER

app.config['SECRET_KEY'] = 'super-secret'
app.config['FLASK_ENV'] = 'development'  # Set Flask environment to development
app.debug = app.config['FLASK_ENV'] == 'development' # Enable Flask debug mode

## Create folders if they don't exist
if not os.path.exists(app.config['MODEL_FOLDER']):
    os.makedirs(app.config['MODEL_FOLDER'])

if not os.path.exists(app.config['UPLOAD_FOLDER_TEST']):
    os.makedirs(app.config['UPLOAD_FOLDER_TEST'])


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
