import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import hw0_318170917_322995358_eval
import matplotlib.pyplot as plt
import numpy as np

input_size = 784  # 28*28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


def newsoftmax(x):
    max = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - max)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    return x_exp / x_exp_sum


def relu(out):
    zeros_vec = torch.zeros(out.shape)
    return torch.maximum(out, zeros_vec)


def Cross_entropy_loss(outputs, labels):
    return (-1 * torch.log(outputs).gather(1, labels.unsqueese(1))).sum()


# define the model
class TwoLayers(nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayers, self).__init__()
        self.linear1 = nn.Linear(input_size, 100)
        self.linear2 = nn.Linear(100, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = relu(out)
        out = self.linear2(out)
        return out


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.3081,],
                                                         std=[0.1306,])])

    train_dataset = dsets.MNIST(root='./data', train=True, transform=transform,
                                download=True)

    # Dataset Loader (Input Pipline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    model = TwoLayers(input_size, num_classes)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    acc_train = []
    acc_test = []
    for epoch in range(num_epochs):
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28)
            optimizer.zero_grad()
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (predicted == labels).sum()
        torch.save(model, 'model.pkl')
        acc_test.append(hw0_318170917_322995358_eval.evaluate_hw0())
        acc_train.append(correct / total)
    plt.plot(np.arange(num_epochs), acc_train, label="train")
    plt.plot(np.arange(num_epochs), acc_test, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Compering test and train accuracy")
    plt.show()