from utils.get_predictions_vision import get_predictions_vision
import torch
from os import getcwd,listdir
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import cv2
from sklearn.preprocessing import LabelEncoder
import PIL.Image as Image
from pylab import rcParams
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import models
from collections import defaultdict
from utils import (
                  MemesDataset,
                  train_vision_model,
                  show_confusion_matrix,
                  get_test_predictions_vision,
                  inception_model
)
import os

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


Path_To_Save_Df = 'your_path'
Path_To_Save_Model = 'your_path'

## Importing Dataset

curr_dir = getcwd()
dataset_dir = join(curr_dir,'drive','MyDrive','Codalab','Meme Classification Challenge','Dataset')
train_img_dir_path = join(dataset_dir,'train_img_dir')
train_df_path = join(dataset_dir,'train.csv')
test_img_dir_path = join(dataset_dir,'test_img_dir')
test_df_path = join(dataset_dir,'test.csv')


## configuring our setup

sns.set(style='darkgrid', palette='muted', font_scale=1.2)

COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(COLORS_PALETTE))

rcParams['figure.figsize'] = (16, 12)



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

### Dataset Inspection

print(f" No. of training images : {len(listdir(train_img_dir_path))}")
print(f" No. of test images: {len(listdir(test_img_dir_path))}")

## reading the train_df
train_df = pd.read_csv(train_df_path)
train_df.sample(10)

## Conversion of Labels
le = LabelEncoder()
train_df['labels'] = le.fit_transform(train_df['labels'])

# class names of our labels
class_names = ['Not_troll','troll']
label_dict = {0:'Not_troll',1:'troll'}

## saving the final train_df 
train_df.to_csv(join(dataset_dir,'final_train.csv'),index = False)
## reading the final train_df
final_train = pd.read_csv(join(dataset_dir,'final_train.csv'))

## reading the test_df
test_df = pd.read_csv(test_df_path)
test_df.sample(10)
## conversion of labels
test_df['labels'] = le.fit_transform(test_df['label'])

## saving the final test_df
test_df.to_csv(join(dataset_dir,'final_test.csv'),index = False)


## Data Transforms

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
img_size = 299

train_transforms = T.Compose([
                            T.RandomResizedCrop(size=img_size),
                            T.ToTensor(),
                            T.Normalize(mean_nums, std_nums)
                  ])

test_transforms =  T.Compose([
                            T.RandomResizedCrop(size=img_size),
                            T.ToTensor(),
                            T.Normalize(mean_nums, std_nums)
                ])

## Loading our datasets

train_dataset = MemesDataset(join(dataset_dir,'final_train.csv'),train_img_dir_path,train_transforms)
test_dataset = MemesDataset(join(dataset_dir,'final_test.csv'),test_img_dir_path,test_transforms)

## Creating dataloaders

batch_size = 128
validation_split = .2
shuffle_dataset = True

# Creating data indices for training and validation splits:
train_dataset_size = len(train_dataset)
indices = list(range(train_dataset_size))
split = int(np.floor(validation_split * train_dataset_size))

if shuffle_dataset :
    np.random.seed(seed_val)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


## Defining our model
model = inception_model(
    num_classes = 2,
    feature_extract = True,
    use_pretrained = True
)

model.to(device)

train_size = len(train_dataset) - split 
val_size = split

print(f'The training set size is {train_size} and validation set size is {val_size}')

## Model training

model, history = train_vision_model(model, train_loader,validation_loader, train_size,val_size, device)

## Performance on training set

y_pred, y_test = get_predictions_vision(model, train_loader)
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
show_confusion_matrix(cm, class_names)

## Performance on validation set

y_pred, y_test = get_predictions_vision(model, validation_loader)
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
show_confusion_matrix(cm, class_names)

## Performance on test set

y_pred, y_test = get_predictions_vision(model, test_loader)
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
show_confusion_matrix(cm, class_names)

## Saving the model

torch.save(model,Path_To_Save_Model)



"""## Getting test set predictions"""

y_ids,y_preds = get_test_predictions_vision(model, test_loader)

sns.countplot(x = le.inverse_transform(y_preds))

df = pd.DataFrame({
    'id':y_ids,
    'label':le.inverse_transform(y_preds)
})


df.to_csv(join(Path_To_Save_Df,'submission.csv'),index = False)

