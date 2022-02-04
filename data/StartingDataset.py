import torch
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import numpy as np

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    def __init__(self):
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#
        
        self.images = pd.read_csv("cassava-leaf-disease-classification/train.csv")

        # print(self.images.info())
        # print(len(self.images))

    def __getitem__(self, index):
        label = self.images.loc[index, 'label']

        im = Image.open("cassava-leaf-disease-classification/train_images/" + self.images.loc[index ,'image_id'])
        trans1 = transforms.ToTensor()
        image_tensor = trans1(im)
        # TODO: resize to match rest of program
        # image_tensor = trans1(im).reshape([3, 224, 224])

        return image_tensor, label

    def __len__(self):
        return len(self.images)

