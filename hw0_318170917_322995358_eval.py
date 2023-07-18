import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

input_size = 784  # 28*28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

def evaluate_hw0():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.3081, ],
                                                         std=[0.1306, ])])



    test_dataset = dsets.MNIST(root='./data', train=False, transform=transform)

    # Dataset Loader (Input Pipline)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # normalizing the data
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=transforms.ToTensor())

    model= torch.load('model.pkl')
    total = 0.0
    correct = 0.0
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)
        outputs = model(images)
        predicted = torch.argmax(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
    return (correct/total).item()