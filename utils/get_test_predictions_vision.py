import torch 

def get_test_predictions_vision(model, data_loader,device):
      
  model = model.eval()

  predictions , imagenames = [] , []
    
  print('Predicting on test dataset')
    
  with torch.no_grad():
    
    for idx , (b_imagenames,b_images,b_captions) in enumerate(data_loader):
    
      print(f'Predicting on batch {idx + 1}')
      
      inputs = b_images.to(device)

      outputs = model(inputs)

      _, preds = torch.max(outputs, 1)
          
      imagenames.extend(b_imagenames)
      
      predictions.extend(preds)
      
  predictions = torch.as_tensor(predictions).cpu()

  return (imagenames , predictions)
