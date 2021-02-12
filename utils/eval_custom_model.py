import time 
import torch
from utils import  good_update_interval,format_time,check_accuracy
def eval_custom_model(model,py_inputs, py_attn_masks, py_labels,criterion):
    # Prediction on test set
    t0 = time.time()

    print('Doing validation on {:,} sentences...'.format(len(py_labels)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Choose an interval on which to print progress updates.
    update_interval = good_update_interval(total_iters=len(py_inputs), num_desired_updates=10)

    # Measure elapsed time.
    t0 = time.time()
    total_val_loss = 0

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
        b_labels = py_labels[step].to(device)
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            logits = model(b_input_ids, b_input_mask)


        loss = criterion(logits,b_labels)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

        total_val_loss += loss.item()
    
    val_accuracy = check_accuracy(predictions,true_labels)

    validation_time = format_time(time.time() - t0)
    
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_val_loss / len(py_inputs)     
        

    # print('Accuracy: {:.3f}'.format(val_accuracy))
    return (avg_val_loss,val_accuracy,validation_time)
