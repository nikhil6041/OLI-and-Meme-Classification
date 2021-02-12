import torch 
import pandas as pd 
import os 
import PIL.Image as Image

class MemesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir_path, transform=None,flg = True):
        ## flag for training or test (default true for training set)
        self.flg = flg
        ## reading the dataframe
        self.dataframe = pd.read_csv(csv_file)
        ## getting the root_dir_path
        self.root_dir_path = root_dir_path
        ## getting imageids from the dataframe
        self.imagenames = self.dataframe["imagename"]
        # getting all captions from the dataframe
        self.captions = self.dataframe["captions"]
        ## getting labels from the dataset if the dataset is training type
        if self.flg:
          self.labels = self.dataframe["labels"]
        ## getting possible transformations of the images
        self.transform = transform


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        ## get the imagename from the dataframe
        imagename = self.imagenames[index]
        ## get the path of the image        
        img_path = os.path.join(self.root_dir_path,imagename)
        ## open the image using PIL.Image and convert it to RGB in case of an extra alpha channel
        image = Image.open(img_path).convert('RGB')
        ## get caption corresponding to the given image
        caption = self.captions[index]
        ## applying transformations to image
        if self.transform:
          image = self.transform(image)
        ## return y label if dataset type is training
        if self.flg:
          y_label = torch.tensor(int(self.labels[index]))
          return (image,caption ,y_label)

        return (image,caption)