import torch 
def get_predictions_vision(model, data_loader,device):
    model = model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for batch_idx ,(b_images,b_captions, b_labels) in enumerate(data_loader):
            print(f' Processing batch {batch_idx+1} ')
            b_images = b_images.to(device)
            b_labels = b_labels.to(device)

            outputs = model(b_images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds)
            real_values.extend(b_labels)
    
    predictions = torch.as_tensor(predictions).cpu()
    real_values = torch.as_tensor(real_values).cpu()
    
    return predictions, real_values
