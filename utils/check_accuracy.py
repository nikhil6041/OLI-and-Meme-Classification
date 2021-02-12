import numpy as np
def check_accuracy(predictions,true_labels):
    """
    Used for checking accuracy across each epoch
    """
    # Combine the results across the batches.
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Choose the label with the highest score as our prediction.
    preds = np.argmax(predictions, axis=1).flatten()

    # Calculate simple flat accuracy -- number correct over total number.
    accuracy = (preds == true_labels).mean()

    return accuracy