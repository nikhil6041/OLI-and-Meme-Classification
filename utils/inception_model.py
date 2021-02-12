import torchvision.models as models
import torch.nn as nn
from utils import set_parameter_requires_grad

def inception_model(
    num_classes = 2,
    feature_extract = True,
    use_pretrained = True
):
    model = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)


    # Handle the auxilary net
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)


    # Handle the primary net
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)

    return model 