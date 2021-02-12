import torchvision.models as models
import torch.nn as nn
from utils import set_parameter_requires_grad

def resnet50_model(
    num_classes = 2,
    feature_extract = True,
    use_pretrained = True
):
    
    model = models.resnet50(use_pretrained)
    set_parameter_requires_grad(model, feature_extract)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)

    return model 