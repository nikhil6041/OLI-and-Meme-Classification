import torch
import torch.nn as nn

def nll_loss_with_class_weights(class_wts,device):
    # convert class weights to tensor
    weights= torch.tensor(class_wts,dtype=torch.float)
    weights = weights.to(device)
    return nn.NLLLoss(weight=weights)

