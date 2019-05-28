import numpy as np
import pandas as pd

import torch
import torch.nn as nn  # Neural network
import torch.optim as optim  # Various optimization algorithms

from torch.utils.data import DataLoader  # Abstract class representing Dataset
from torchvision import transforms  # Common image transformations

from utils.data_set import FacialKeyPointsDataset
from utils.rescale import Rescale
from utils.random_crop import RandomCrop
from utils.normalize import Normalize
from utils.to_tensor import ToTensor

from model.naimishnet import NaimishNet


def train_net(neural_net, n_epochs, criterion, train_loader, optimizer_type, lr):
    print(neural_net)
    if optimizer_type == "Adam":
        opt = optim.Adam(params=neural_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    elif optimizer_type == "SGD":
        opt = optim.SGD(params=neural_net.parameters(), lr=lr)
    elif optimizer_type == "RMS":
        opt = optim.RMSprop(params=neural_net.parameters(), lr=0.01)
    else:
        raise Exception("Unknown optimizer type!")

    # Prepare the net for training
    neural_net.train()

    for epoch in range(n_epochs):  # Loop over the dataset multiple items

        running_loss = 0.0

        # Train on batches of data, train_loader
        for batch_i, data in enumerate(train_loader):
            # Get the image and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']
            # images, key_pts = images.to(device), key_pts.to(device)

            # Flatten points
            key_pts = key_pts.view(key_pts.size(0), -1)

            # Convert variables to floats for regressin loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            # Forward pass to get outputs
            output_pts = neural_net(images)

            # Calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # Zero the parameters(weights) gradients
            opt.zero_grad()

            # Backward pass to calculate the weight gradients
            loss.backward()

            # Update the weights
            opt.step()

            # Print loss statistics to convert loss into a scalar and add it to the
            # running_loss, use .item()
            running_loss += loss.item()

            if batch_i % 10 == 9:  # Print every 10 batches
                print("Epoch: {}, Batch: {}, Avg. Loss: {}".format(epoch + 1, batch_i + 1,
                                                                   running_loss / 1000))
                
                # Write parameters to a csv file
                param_list = np.array([[epoch+1, batch_i+1, running_loss, lr]])
                df = pd.DataFrame(param_list, columns=['Epoch', 'Batch', 'Loss', 'LR'])
                df.to_csv('param_file.csv', mode='a', header='False')
                
                running_loss = 0.0

            print("Training has been finished!")


def test_multiple_setups(n_epochs, batches, optimizer_types, learning_rates):
    # Chain transformations using Compose
    data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
    transformed_dataset = FacialKeyPointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    neural_net = NaimishNet()
    neural_net.to(device)

    # Use a for loop to test and see how the variables influence the model accuracy
    for batch in batches:
        data_loader = DataLoader(transformed_dataset, batch_size=batch, shuffle=True, num_workers=4)
        for optimizer in optimizer_types:
            for lr in learning_rates:
                print("Batch: ", batch)
                print("Optimizer: ", optimizer)
                print("Learning rate: ", lr)
                print("NNet: ", "NaimishNet")

                train_net(neural_net, n_epochs, nn.MSELoss(), data_loader, optimizer, lr)


def train_and_save(n_epoch, batch_size, optimizer, learning_rate, path):
    data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
    transformed_dataset = FacialKeyPointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    data_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # Alternatif:
    neural_net = NaimishNet()
    neural_net.to(device)

    train_net(neural_net, n_epoch, nn.MSELoss(), data_loader, optimizer, learning_rate)
    torch.save(neural_net.state_dict(), path)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    neural_net = NaimishNet()
    # Try various combinations. (epoch, batches, optimizer type, learning rate)
    # test_multiple_setups(1, [8, 12, 32, 64], ["Adam", "SGD", "RMS"], [0.001, 0.005, 0.01])
    train_and_save(1, 64, "SGD", 0.1, "./saved_models/mymodel.pt")
