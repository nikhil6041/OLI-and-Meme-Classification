import torch 
from torch import nn, optim
from collections import defaultdict
from torch.optim import lr_scheduler
from utils import train_epoch ,eval_vision_model
def train_vision_model(
    model,
    train_data_loader,
    val_data_loader, 
    train_dataset_size,
    val_dataset_size,
    device, 
    n_epochs=3):
  
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(n_epochs):

        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
                                    model, 
                                    train_data_loader, 
                                    criterion, 
                                    optimizer, 
                                    device, 
                                    scheduler, 
                                    train_dataset_size
                                )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_vision_model(
                                    model, 
                                    val_data_loader, 
                                    criterion, 
                                    device, 
                                    val_dataset_size
                            )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    print(f'Best val accuracy: {best_accuracy}')
    
    model.load_state_dict(torch.load('best_model_state.bin'))

    return model, history
