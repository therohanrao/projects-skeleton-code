import torch
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import random

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    def __init__(self, iseval, transform=None):
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#

        self.iseval = iseval
        self.transform = transform

        
        preDF = pd.read_csv("cassava-leaf-disease-classification/train.csv")
        
        if self.iseval:
            self.images = preDF.truncate(after = len(preDF)/5).reset_index(drop = True)
            print(self.images.info())
        else:
            self.images = preDF.truncate(before = len(preDF)/5).reset_index(drop = True)
            print(self.images.info())

    
    def __getitem__(self, index):
        label = self.images.loc[index, 'label']

        im = Image.open("cassava-leaf-disease-classification/train_images/" + self.images.loc[index ,'image_id'])
        im = im.resize((224, 224))

        augment = random.choice([0,1])
        if augment:
            image_tensor = self.transform(im)
        else:
            toTensor = transforms.ToTensor()
            image_tensor = toTensor(im)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            normalize(image_tensor)

        return image_tensor, label

    def __len__(self):
        return len(self.images)

