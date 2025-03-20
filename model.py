import torch
import torch.nn as nn

class MNIST_MLP(nn.Module):
    def __init__(self, n_classes=10):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MNIST_CNN(nn.Module):

    def __init__(self, n_classes=10):
        super(MNIST_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Input: (1, 28, 28), Output: (32, 26, 26)
        self.pool = nn.MaxPool2d(2, 2)  # Kernel size 2, Stride 2, reduces size by half
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Input: (32, 13, 13), Output: (64, 11, 11)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 5 * 5, n_classes)  # Flattened to 1600 dimensions for the input to fc1

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (batch_size, 32, 26, 26) -> (batch_size, 32, 13, 13)
        x = self.pool(torch.relu(self.conv2(x)))  # (batch_size, 64, 11, 11) -> (batch_size, 64, 5, 5)
        
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64 * 5 * 5)
        x = self.fc1(x)  # (batch_size, 128)
        return x

class CIFAR_CNN(nn.Module):
    def __init__(self, n_classes=10):
        super(CIFAR_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def reset_parameters(self):
        for layer in self.conv_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.fc_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
