import torch 
''''
This function is for training the inception model
'''
def train_epoch(
  model, 
  data_loader, 
  criterion, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  print(f'Doing training on {n_examples} samples')
  for batch_idx , (b_images,b_captions ,labels) in enumerate(data_loader):
    print(f' Processing batch {batch_idx+1} ')
    b_images = b_images.to(device)
    labels = labels.to(device)

    outputs, aux_outputs = model(b_images)
    loss1 = criterion(outputs, labels)
    loss2 = criterion(aux_outputs, labels)
    # outputs = model(inputs)
    # loss = criterion(outputs,labels)
    loss = loss1 + 0.4*loss2
    _, preds = torch.max(outputs, 1)
    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())
  
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  scheduler.step()

  return correct_predictions.double() / n_examples, np.mean(losses)
