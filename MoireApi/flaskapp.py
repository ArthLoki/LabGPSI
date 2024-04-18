from flask import Flask, render_template, request, jsonify
from flask import flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from moire.test_04 import load_and_evaluate

import os

UPLOAD_FOLDER_TRAIN = 'uploads/images/train'
UPLOAD_FOLDER_TEST = 'uploads/images/test'
MODEL_FOLDER = 'uploads/savedmodels'

ALLOWED_EXTENSIONS_FILE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_MODEL = {'h5', 'keras'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER_TEST'] = UPLOAD_FOLDER_TEST
app.config['UPLOAD_FOLDER_TRAIN'] = UPLOAD_FOLDER_TRAIN
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['SECRET_KEY'] = 'super-secret'
app.config['FLASK_ENV'] = 'development'  # Set Flask environment to development
app.debug = True  # Enable Flask debug mode
# app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)

## Create folders if they don't exist
if not os.path.exists(app.config['MODEL_FOLDER']):
    os.makedirs(app.config['MODEL_FOLDER'])

if not os.path.exists(app.config['UPLOAD_FOLDER_TRAIN']):
    os.makedirs(app.config['UPLOAD_FOLDER_TRAIN'])
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], 'positive'))
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], 'negative'))

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
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            # return redirect(request.url)
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # CHECK MODEL FILE

        if len(os.listdir(app.config['MODEL_FOLDER'])) == 0:
            # check if the post request has the file part
            saved_model = request.files['savedmodel']
            if saved_model.filename == '':
                return jsonify({'error': 'No selected model'})

            if saved_model and allowed_file(saved_model.filename, 1):
                modelname = secure_filename(str(saved_model.filename))
                if not os.path.exists(os.path.join(app.config['MODEL_FOLDER'], modelname)):
                    saved_model.save(os.path.join(app.config['MODEL_FOLDER'], modelname))
            else:
                return jsonify({'error': 'Model file not allowed'})
        else:
            # saved_model = None
            modelname = os.listdir(app.config['MODEL_FOLDER'])[0] if os.listdir(app.config['MODEL_FOLDER']) else None
            # print(modelname)

        # CHECK IMAGE FILE

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            # flash('No selected file')
            # return redirect(request.url)
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename, 0) and modelname is not None:
            filename = secure_filename(str(file.filename))
            # modelname = os.listdir(app.config['UPLOAD_FOLDER_TEST'])[0]

            file.save(os.path.join(app.config['UPLOAD_FOLDER_TEST'], filename))
            # return redirect(url_for('download_file', name=filename))

            # chamada do moire
            modelpath = str(os.path.join(app.config['MODEL_FOLDER'], modelname)).replace('/', '\\')

            if len(os.listdir(app.config['UPLOAD_FOLDER_TEST'])) > 0:
                status = load_and_evaluate(
                    modelpath,
                    os.path.join(app.config['UPLOAD_FOLDER_TEST']))

                if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER_TEST'], filename)):
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER_TEST'], filename))

                return jsonify({'filename': filename, 'status': status, 'modelname': modelname})
            return jsonify({'error': 'No file found'})

            # return render_template(
            #     "response.html",
            #     filename=filename.upper(),
            #     loss=status.get('loss'),
            #     accuracy=status.get('accuracy'),
            #     cm=status.get('confusion_matrix'),
            #     cr=status.get('classification_report'),
            #     # file=url_for('download_file', name=filename)
            # )

    return render_template('index.html', empty_model_folder=len(os.listdir(app.config['UPLOAD_FOLDER_TEST'])) == 0)


# @app.route('/uploads/<name>')
# def download_file(name):
#   return send_from_directory(app.config["UPLOAD_FOLDER_TEST"], name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
