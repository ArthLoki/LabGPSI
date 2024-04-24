import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

WIDTH = 750
HEIGHT = 1000
TRANSFORM_COUNT = 12  # Número de transformações por imagem original

def load_and_evaluate(model_path, test_dir):

    img_path = os.path.join(test_dir, os.listdir(test_dir)[0])

    # Load the trained model
    try:
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        return {'Error': f'Error loading {model_path.split("/")[-1]} in test_07.py:\n{e}'}

    # Preprocessing image
    try:
        print("Preprocessing image...")
        # img = image.load_img(os.path.join(test_dir, os.listdir(test_dir)[0]), target_size=(WIDTH, HEIGHT))
        # img = image.img_to_array(img)
        # img = np.expand_dims(img, axis=0)
        # img /= 255.0  # Normalize pixel values

        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((WIDTH, HEIGHT))
        img = np.expand_dims(np.array(img), axis=0)
        img = np.expand_dims(img, axis=-1)  # Add a singleton dimension for channels
        img = img.astype('float32') / 255.0  # Normalize pixel values

        print("Preprocess image successfully!")
    except Exception as e:
        return {'Error': f'Error preprocessing {os.listdir(test_dir)[0]} in test_07.py:\n{e}'}

    # Predictions
    try:
        print("Predicting...")
        predictions = model.predict(img)
        print("Predictions successfully!")
    except Exception as e:
        return {'Error': f'Error in generation of predictions for {os.listdir(test_dir)[0]} in test_07.py:\n{e}'}

    # # Generating array of predictions
    # predictions = np.round(predictions).astype(int).flatten()

    # Assuming binary classification (1: non-moire, 0: moire)
    print('-----> predictions:', predictions[0][0])
    if predictions[0][0] < 0.5:
        # print(f'{img_path}: Moiré pattern detected')
        return {'status': 'Moire pattern detected'}
    else:
        # print(f'{img_path}: No moiré pattern detected')
        return {'status': 'No Moire pattern detected'}