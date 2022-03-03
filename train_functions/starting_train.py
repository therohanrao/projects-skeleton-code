import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 8
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 8
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), 0.001)
    #optim.Adam([var1, var2], lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    # Use GPU
    if torch.cuda.is_available(): # Check if GPU is available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.train()

    step = 1

    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        # for batch in train_loader:
        loop = tqdm(train_loader)
        for batch in loop:
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model.forward(images) # Same thing as model.forward(images) = model(images)

            # Backpropagation and gradient descent
            # Compute validation loss and accuracy.
            # Log the results to Tensorboard.
            # Don't forget to turn off gradient calculations!
            # Backprop
            loss = loss_fn(outputs, labels)
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad() # Clear gradients before next iteration

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                print ("\nevaluating...\n")
                evaluate(val_loader, model, loss_fn)
            #print('Epoch:', epoch + 1, 'Loss:', loss.item())
            loop.set_postfix({"loss": f"{loss.item() : .03f}"})
            step += 1

        lr_scheduler.step()

        


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    if torch.cuda.is_available(): # Check if GPU is available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Computes the loss and accuracy of a model on the validation dataset.
    correct = 0
    total = 0
    with torch.no_grad(): # IMPORTANT: turn off gradient computations
        for batch in tqdm(val_loader):
            # for batch in val_loader:
                #batch = next(iter(val_loader))      
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
      
                #images = torch.reshape(images, (-1, 1, 224, 224))
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
      
                # labels == predictions does an elementwise comparison
                # e.g.                labels = [1, 2, 3, 4]
                #                predictions = [1, 4, 3, 3]
                #      labels == predictions = [1, 0, 1, 0]  (where 1 is true, 0 is false)
                # So the number of correct predictions is the sum of (labels == predictions)
                correct += (labels == predictions).int().sum()
                total += len(predictions)

    print('Accuracy:', (correct / total).item())

    model.train()
