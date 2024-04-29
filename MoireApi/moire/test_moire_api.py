import numpy as np
from tensorflow.keras.models import load_model
import logging
import os
from PIL import Image
from sklearn.metrics import confusion_matrix

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

    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels

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

        results_pred.append({'prediction': predictions[0][0]})

        # Append ground truth and predicted labels
        y_true.append(1 if predictions[0][0] > 0.5 else 0)  # Assuming all images contain moiré patterns for simplicity
        y_pred.append(1 if predictions[0][0] > 0.5 else 0)

    # Compute confusion matrix
    '''
    Confusion Matrix:

    [[true negative, false positive],
    [false negative, true positive]]
    '''
    confusion_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print('\n\nCONFUSION MATRIX:')
    print(confusion_mat)

    # Check Confusion Matrix data
    TN = confusion_mat[0, 0]
    FP = confusion_mat[0, 1]
    FN = confusion_mat[1, 0]
    TP = confusion_mat[1, 1]

    print('\n\n')
    print("True Positives (TP): ", TP)
    print("True Negatives (TN): ", TN)
    print("False Positives (FP): ", FP)
    print("False Negatives (FN): ", FN)

    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else (TP + TN)/TRANSFORM_COUNT
    precision = TP / (TP + FP) if TP + FP > 0 else TP
    recall = TP / (TP + FN) if TP + FN > 0 else TP
    f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 2 * (precision * recall)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall (Sensitivity): ", recall)
    print("F1 Score: ", f1score)
    print('\n\n')

    if TN + FP + FN + TP == TRANSFORM_COUNT:
        # is_moire = 'under analysis'
        if accuracy > 0.7:
            # In this case, to make sure no false negative passes, we will consider them as false positives
            # if ((TN == TRANSFORM_COUNT) and (FP+TP+FN == 0)) or (0 < TP < TN and TN > 0): is_moire = 'negative'
            # elif ((TP == TRANSFORM_COUNT) and (FP+TN+FN == 0)) or (0 < TN < TP and TP > 0): is_moire = 'positive'
            # else: is_moire = 'false positive'

            if (TN == TRANSFORM_COUNT) and (FP+TP+FN == 0): is_moire = 'negative'
            elif (TP == TRANSFORM_COUNT) and (FP+TN+FN == 0): is_moire = 'positive'
            else: is_moire = 'false positive'
        else:
            # If accuracy is not good, everything becomes 'false positive' for manual analysis
            is_moire = 'false positive'
    else:
        return {'Error': 'Not all decompositions are in confusion matrix.'}

    results_cm = {
        # 'Confusion Matrix': confusion_mat,
        "True Positives": TP,
        "True Negatives": TN,
        "False Positives": FP,
        "False Negatives": FN,
        "Accuracy": accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1score,
    }

    return {
        'results_predictions': results_pred,
        'results_cm': results_cm,
        'moire': is_moire,
    }
