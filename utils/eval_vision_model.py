import torch 
import numpy as np
def eval_vision_model(
    model, 
    data_loader,
    criterion,
    device,
    n_examples):

    model = model.eval()

    losses = []
    correct_predictions = 0

    print(f'Doing validation on {n_examples} samples')

    with torch.no_grad():

        for batch_idx ,(b_images,b_captions, labels) in enumerate(data_loader):

            print(f' Processing batch {batch_idx+1} ')

            b_images = b_images.to(device)
            labels = labels.to(device)

            outputs = model(b_images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)
