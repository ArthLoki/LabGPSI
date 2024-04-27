import numpy as np
from tensorflow.keras.models import load_model
import logging
import os
from PIL import Image

WIDTH = 750
HEIGHT = 1000
TRANSFORM_COUNT = 12  # Número de transformações por imagem original


def load_and_evaluate(model_path, test_dir):
    logging.basicConfig(level=logging.INFO)

    # Load the trained model
    try:
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        return {'Error': f'Error loading {model_path.split("/")[-1]} in test_moire_api.py:\n{e}'}

    results = []
    results_pred = []
    for i in range(TRANSFORM_COUNT):
        img_path = os.path.join(test_dir, os.listdir(test_dir)[i])

        # Preprocessing image
        try:
            print(f"Preprocessing image {i+1}...")
            test_generator = Image.open(img_path).convert('L')  # Convert to grayscale
            test_generator = test_generator.resize((WIDTH, HEIGHT))
            test_generator = np.expand_dims(np.array(test_generator), axis=0)
            test_generator = np.expand_dims(test_generator, axis=-1)  # Add a singleton dimension for channels
            test_generator = test_generator.astype('float32') / 255.0  # Normalize pixel values

            print(f"Preprocess image {i+1} successfully!")
        except Exception as e:
            return {'Error': f'Error preprocessing {os.listdir(test_dir)[i]} in test_moire_api.py:\n{e}'}

        # Previsões do conjunto de teste
        try:
            logging.info("Gerando previsões para o conjunto de teste...")
            predictions = model.predict(test_generator)
        except Exception as e:
            return {
                'Error': f'Error generating predictions for {model_path.split("/")[-1]} in test_moire_api.py: {e}'}

        # if i not in [4, 8, 12]:
        #     if predictions[0][0] >= 0.03:
        #         moire_prediction = True
        #     else:
        #         moire_prediction = False
        # else:
        #     moire_prediction = False

        # results.append(moire_prediction)
        # results_pred.append({'prediction': predictions[0][0], 'moire': moire_prediction})

        results.append(True if predictions[0][0] > 0.5 else False)
        results_pred.append({'prediction': predictions[0][0], 'moire': True if predictions[0][0] > 0.5 else False})

    count_moire = results.count(True)
    if count_moire == 0:
        is_moire = False
    else:
        is_moire = True

    return {
        'results_predictions': results_pred,
        'moire': is_moire
    }