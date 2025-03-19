import torch
import copy
from tqdm import tqdm

from plot import Plot
from data import MNIST_data_loaders
from model import MNIST_MLP, MNIST_CNN

class Experiment():

    def __init__(self, model, optimizer, criterion, train_loader, val_loader, n_epochs, warmup_epochs, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.device = device

    def sum_weights(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data += source_param.data
        return target

    def normalize_weights(self, model, n):
        for param in model.parameters():
            param.data /= n
        return model

    def train(self):
        self.model.to(self.device)

        batch_train_losses, train_losses, val_losses = [], [], []
        batch_train_accuracies, train_accuracies, val_accuracies = [], [], []

        several_avgs = isinstance(self.warmup_epochs, list)
        n_models = 0 if not several_avgs else [0] * len(self.warmup_epochs)
        avg_model = None if not several_avgs else [None] * len(self.warmup_epochs)

        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            batch_losses = []

            if not several_avgs and epoch >= self.warmup_epochs:
                n_models += 1
                if avg_model is None:
                    avg_model = copy.deepcopy(self.model)
                else:
                    avg_model = self.sum_weights(self.model, avg_model)
            elif several_avgs:
                for i, wu in enumerate(self.warmup_epochs):
                    if epoch >= wu:
                        n_models[i] += 1
                        if avg_model[i] is None:
                            avg_model[i] = copy.deepcopy(self.model)
                        else:
                            avg_model[i] = self.sum_weights(self.model, avg_model[i])
            
            loop = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.n_epochs}')
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                batch_size = len(images)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_train_losses.append(loss.item() / batch_size)
                batch_losses.append(loss.item() / batch_size)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

                batch_acc = pred.eq(labels).sum().item() / labels.size(0)
                batch_train_accuracies.append(100*batch_acc)

                loop.set_postfix(loss=running_loss/total, acc=100*correct/total)

            train_losses.append(sum(batch_losses) / len(batch_losses))
            train_accuracies.append(100*correct / total)
            val_loss, val_acc = self.validate()
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

        if n_models == 0:
            avg_model = self.model
        else:
            avg_model = self.sum_weights(self.model, avg_model)
            avg_model = self.normalize_weights(avg_model, n_models+1)

        return Plot(batch_train_losses, train_losses, val_losses, batch_train_accuracies, train_accuracies, val_accuracies), avg_model


    def validate(self):
        self.model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                val_loss += self.criterion(outputs, labels).item() / len(images)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f'Validation loss: {avg_val_loss}, Accuracy: {100*correct/total:.2f}%')
        return avg_val_loss, 100*correct/total


class MNIST_Experiment(Experiment):

    def __init__(self, config, use_cnn = False):
        self.use_cnn = use_cnn
        self.config = config

        self.model = MNIST_CNN() if use_cnn else MNIST_MLP()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader, self.val_loader, self.test_loader = MNIST_data_loaders(config)
        self.n_epochs = config.num_epochs
        self.warmup_epochs = config.warmup_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
