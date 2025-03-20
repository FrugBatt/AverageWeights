import matplotlib.pyplot as plt
import numpy as np
import os

class MetricsPlot:

    def __init__(self, batch_train_losses, train_losses, val_losses, batch_train_accuracies, train_accuracies, val_accuracies, exp_path, plot_batch_every=50, output_file='training_metrics.png'):
        self.batch_train_losses = batch_train_losses
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.batch_train_accuracies = batch_train_accuracies
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies
        self.exp_path = exp_path
        self.plot_batch_every = plot_batch_every
        self.output_file = output_file

    def save(self):
        plt.clf()

        # Number of epochs and batches
        n_epochs = len(self.train_losses)
        n_batches = len(self.batch_train_losses)

        # Create figure and axis
        fig, ax1 = plt.subplots()

        # Plot losses on the first y-axis (left)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss', color='tab:red')

        # Create x-axis for batch-wise and epoch-wise
        x_batches = np.arange(0, n_batches, self.plot_batch_every) # x-axis for batch-wise
        x_epochs = np.linspace(0, n_batches, n_epochs) # x-axis for epoch-wise
        
        # Plot batch-wise and epoch-wise loss
        ax1.plot(x_batches, self.batch_train_losses[::self.plot_batch_every], label='Batch Train Loss', color='tab:red', alpha=0.7)
        ax1.plot(x_epochs, self.train_losses, label='Epoch Train Loss', color='tab:orange', linestyle='--')
        ax1.plot(x_epochs, self.val_losses, label='Epoch Validation Loss', color='tab:purple', linestyle='--')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Create a second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='tab:blue')

        # Plot batch-wise and epoch-wise accuracy
        ax2.plot(x_batches, self.batch_train_accuracies[::self.plot_batch_every], label='Batch Train Accuracy', color='tab:blue', alpha=0.7)
        ax2.plot(x_epochs, self.train_accuracies, label='Epoch Train Accuracy', color='tab:green', linestyle='--')
        ax2.plot(x_epochs, self.val_accuracies, label='Epoch Validation Accuracy', color='tab:cyan', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Title and layout
        plt.title('Training and Validation Loss & Accuracy')
        plt.tight_layout()
        # plt.show()
        
        # Save plot
        plt.savefig(os.path.join(self.exp_path, self.output_file))

class AvgModelPlot():

    def __init__(self, warmup_epochs, accuracies, test_acc, exp_path, output_file='average_model.png'):
        self.warmup_epochs = warmup_epochs
        self.accuracies = accuracies
        self.test_acc = test_acc
        self.exp_path = exp_path
        self.output_file = output_file

    def save(self):
        plt.clf()
        plt.plot(self.warmup_epochs, self.accuracies, label='Average Model Accuracy')
        plt.axhline(y=self.test_acc, color='r', linestyle='--', label='Unmodified Model Accuracy')
        plt.xlabel('Epoch to start averaging')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of Averaged Models')
        plt.legend()

        plt.savefig(os.path.join(self.exp_path, self.output_file))

class DistribAccuraciesPlot():

    def __init__(self, dist_model_accs, warmup_epochs, dist_avg_model_accs, test_acc, exp_path, output_file='distribution_avg.png'):
        self.dist_model_accs = dist_model_accs
        self.warmup_epochs = warmup_epochs
        self.dist_avg_model_accs = dist_avg_model_accs
        self.test_acc = test_acc
        self.exp_path = exp_path
        self.output_file = output_file

    def save(self):
        plt.clf()

        plt.hist(self.dist_model_accs, bins=20, alpha=0.5, label='Model Accuracies')
        for i in range(len(self.warmup_epochs)):
            plt.hist(self.dist_avg_model_accs[i], bins=20, alpha=0.5, label=f'Avg Model Accuracies starting from epoch {self.warmup_epochs[i]}')
        plt.axvline(x=self.test_acc, color='r', linestyle='--', label='Unmodified Model Accuracy')
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Model Accuracies')
        plt.legend()

        plt.savefig(os.path.join(self.exp_path, self.output_file))

class AvgAccuraciesPlot():

    def __init__(self, avg_inc_accs, exp_path, output_file='avg_increase.png'):
        self.avg_inc_accs = avg_inc_accs
        self.exp_path = exp_path
        self.output_file = output_file

    def save(self):
        plt.clf()

        plt.plot(self.avg_inc_accs, label='Average Increase in Accuracy')
        plt.xlabel('Epoch to start averaging model weights')
        plt.ylabel('Accuracy Increase in Accuracy (%)')
        plt.title('Average Increase in Accuracy of Averaged Models')
        plt.legend()
        
        plt.savefig(os.path.join(self.exp_path, self.output_file))
