import matplotlib.pyplot as plt
import numpy as np

class Plot:

    def __init__(self, batch_train_losses, train_losses, val_losses, batch_train_accuracies, train_accuracies, val_accuracies, plot_batch_every=50):
        self.batch_train_losses = batch_train_losses
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.batch_train_accuracies = batch_train_accuracies
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies
        self.plot_batch_every = plot_batch_every

    def plot_metrics(self):
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
        plt.show()
