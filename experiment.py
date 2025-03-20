import torch
import copy
from torch.utils.data import dataloader
from tqdm import tqdm

from plot import MetricsPlot, AvgModelPlot, DistribAccuraciesPlot,  AvgAccuraciesPlot 
from data import MNIST_data_loaders
from model import MNIST_MLP, MNIST_CNN

class Experiment():

    def __init__(self, model, optimizer, criterion, train_loader, val_loader, test_loader, n_epochs, warmup_epochs, distribution_samples, exp_path, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.distribution_samples = distribution_samples
        self.exp_path = exp_path
        self.device = device

    def run(self):
        if self.distribution_samples is None: # Single experiment
            metrics, avg_models = self.train()
            _, test_acc = self.validate(dataloader=self.test_loader)
            avg_acc = self.avg_accuracy(avg_models, test_acc)
            return metrics, avg_acc
        else: # Distribution of accuracies, no metrics plot
            distrib, avg_acc = self.distribution()
            return distrib, avg_acc


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

        n_models = [0] * len(self.warmup_epochs)
        avg_model = [None] * len(self.warmup_epochs)

        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            batch_losses = []

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

        for i in range(len(self.warmup_epochs)):
            if n_models[i] == 0:
                avg_model[i] = self.model
            else:
                avg_model[i] = self.sum_weights(self.model, avg_model[i])
                avg_model[i] = self.normalize_weights(avg_model[i], n_models[i]+1)

        return MetricsPlot(batch_train_losses, train_losses, val_losses, batch_train_accuracies, train_accuracies, val_accuracies, self.exp_path), avg_model


    def validate(self, model=None, dataloader = None):
        if model is None:
            model = self.model
        if dataloader is None:
            dataloader = self.val_loader

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                val_loss += self.criterion(outputs, labels).item() / len(images)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        avg_val_loss = val_loss / len(dataloader)
        print(f'Validation loss: {avg_val_loss}, Accuracy: {100*correct/total:.2f}%')
        return avg_val_loss, 100*correct/total
    
    def avg_accuracy(self, avg_models, test_acc):
        avg_accs = []
        for i in range(len(avg_models)):
            print(f'Average model starting from epoch {self.warmup_epochs[i]}')
            _, acc = self.validate(model=avg_models[i], dataloader=self.test_loader)
            avg_accs.append(acc)
        return AvgModelPlot(self.warmup_epochs, avg_accs, test_acc, self.exp_path)
    
    def distribution(self):
        avg_inc_accs = [0.0] * len(self.warmup_epochs)
        dist_model_accs = []
        dist_avg_model_accs = [[] for _ in range(len(self.warmup_epochs))]

        for _ in range(self.distribution_samples):
            self.model.reset_parameters()

            _, avg_models = self.train()

            test_loss, test_acc = self.validate(dataloader=self.test_loader)
            dist_model_accs.append(test_acc)

            for i in range(len(self.warmup_epochs)):
                print(f'Average model starting from epoch {self.warmup_epochs[i]}')
                _, avg_acc = self.validate(model=avg_models[i], dataloader=self.test_loader)
                dist_avg_model_accs[i].append(avg_acc)
                inc_acc = avg_acc - test_acc
                avg_inc_accs[i] += inc_acc

        avg_inc_accs = [acc / self.distribution_samples for acc in avg_inc_accs]

        mean_model_accs = sum(dist_model_accs) / self.distribution_samples

        return DistribAccuraciesPlot(dist_model_accs, self.warmup_epochs, dist_avg_model_accs, mean_model_accs, self.exp_path), AvgAccuraciesPlot(avg_inc_accs, self.exp_path)



class MNIST_Experiment(Experiment):

    def __init__(self, config, use_cnn = False):
        self.model = MNIST_CNN() if use_cnn else MNIST_MLP()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader, self.val_loader, self.test_loader = MNIST_data_loaders(config)
        self.n_epochs = config.num_epochs
        self.warmup_epochs = config.warmup_epochs
        self.distribution_samples = config.distribution_samples
        self.exp_path = config.exp_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
