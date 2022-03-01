import os

import constants
import torch
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from networks.StartingNetwork import StartingNetwork2
from train_functions.starting_train import starting_train
import torchvision.transforms as transforms


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    transform_train = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform_val = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset(0, transform = transform_train)
    val_dataset = StartingDataset(1, transform = transform_val)
    model = StartingNetwork2()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )

if __name__ == "__main__":
    main()
