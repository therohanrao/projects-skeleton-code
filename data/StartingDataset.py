import torch
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import numpy as np

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    def __init__(self, iseval):
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#
        

        preDF = pd.read_csv("cassava-leaf-disease-classification/train.csv")
        
        if iseval:
            self.images = preDF.truncate(after = len(preDF)/5).reset_index(drop = True)
            print(self.images.info())
        else:
            self.images = preDF.truncate(before = len(preDF)/5).reset_index(drop = True)
            print(self.images.info())

    
    def __getitem__(self, index):
        label = self.images.loc[index, 'label']

        im = Image.open("cassava-leaf-disease-classification/train_images/" + self.images.loc[index ,'image_id'])
        im = im.resize((224, 224))
        trans1 = transforms.ToTensor()
        image_tensor = trans1(im)

        return image_tensor, label

    def __len__(self):
        return len(self.images)

