from sklearn.metrics import confusion_matrix


def getCM(y_true, y_pred):
    try:
        # Compute confusion matrix
        '''
        Confusion Matrix:

        [[true negative, false positive],
        [false negative, true positive]]
        '''

        confusion_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Check Confusion Matrix data
        TN = confusion_mat[0, 0]
        FP = confusion_mat[0, 1]
        FN = confusion_mat[1, 0]
        TP = confusion_mat[1, 1]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results_cm = {
            # 'Confusion Matrix': confusion_mat,
            "True Positive": TP,
            "True Negative": TN,
            "False Positive": FP,
            "False Negative": FN,
            "Accuracy": accuracy,
            'Precision': precision,
            'Recall (Sensitivity)': recall,
            'F1 Score': f1score,
        }

        return results_cm
    except Exception as e:
        return {'Error': 'getCM error: ' + str(e)}


def count_boolean_channels(results):
    # Create counter variable
    channels_boolean_counter = {
        "LL": {True: 0, False: 0},
        "LH": {True: 0, False: 0},
        "HL": {True: 0, False: 0},
        "HH": {True: 0, False: 0},
    }

    # Increment counter variable
    for key, value in results.items():
        channels_boolean_counter[key][value] += 1

    return channels_boolean_counter