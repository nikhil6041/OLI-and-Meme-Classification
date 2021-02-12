import time 
import torch
import numpy as np
from utils import  good_update_interval,format_time
def get_predictions_on_test(py_inputs, py_attn_masks,py_ids):
    '''
    This function is being used for making predictions on dataset with labels 
    '''
    
    print('Predicting labels for {:,} test batches...'.format(len(py_inputs)))

  # Put model in evaluation mode
    model.eval()

  # Tracking variables 
    predictions , true_labels , ids = [], [] , []

  # Choose an interval on which to print progress updates.
    update_interval = good_update_interval(total_iters=len(py_inputs), num_desired_updates=10)

  # Measure elapsed time.
    t0 = time.time()

  # Put model in evaluation mode
    model.eval()

  # For each batch of training data...
    for step in range(0, len(py_inputs)):

      # Progress update every 100 batches.
        if step % update_interval == 0 and not step == 0:
          # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
          
          # Calculate the time remaining based on our progress.
            steps_per_sec = (time.time() - t0) / step
            remaining_sec = steps_per_sec * (len(py_inputs) - step)
            remaining = format_time(remaining_sec)

          # Report progress.
            print('  Batch {:>7,}  of  {:>7,}.    Elapsed: {:}.  Remaining: {:}'.format(step, len(py_inputs), elapsed, remaining))

      # Copy the batch to the GPU.
        b_input_ids = py_inputs[step].to(device)
        b_input_mask = py_attn_masks[step].to(device)
      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, b_input_mask)

        logits = outputs.logits 

      # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        ids.append(py_ids[step])
      # Store predictions
        predictions.append(logits)
      # Combine the results across the batches.
    predictions = np.concatenate(predictions, axis=0)
    ids = np.concatenate(ids,axis = 0)
  # Choose the label with the highest score as our prediction.
    preds = np.argmax(predictions, axis=1).flatten()

    return (ids,preds)
