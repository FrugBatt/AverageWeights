import argparse
import os

def parse_to_int_list(string):
    return [int(x) for x in string.split(',')]

class ExperimentConfig:

    def __init__(self,
        task,
        num_epochs,
        warmup_epochs,
        batch_size,
        distribution_samples,
        lr,
        lr_end,
        exp_path
    ):
        self.task = task
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.distribution_samples = distribution_samples
        self.lr = lr
        self.lr_end = lr_end
        self.exp_path = exp_path

def load_config():
    parser = argparse.ArgumentParser(description='Average Weights Experiments')
    parser.add_argument('--task', type=str, default='mnist', help='Task to run')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--warmup_epochs', type=str, default=5, help='Number of warmup epochs (for the fine-tuned average)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--distribution_samples', type=int, default=None, help='Number of samples for the distribution of accuracies (Need to be specified for the distribution plot)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (initial learning rate if scheduler is used)')
    parser.add_argument('--lr_end', type=float, default=None, help='Enable the learning rate linear scheduler and final learning rate')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment (for saving the model and plot)')

    args = parser.parse_args()

    warmup_epochs = parse_to_int_list(args.warmup_epochs)
    # if len(warmup_epochs) == 1:
    #     warmup_epochs = warmup_epochs[0]

    if args.experiment_name is None:
        exp_path = os.path.join('experiments', args.task)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        else:
            raise ValueError('Experiment with the same name already exists')
    else:
        exp_path = os.path.join('experiments', args.experiment_name)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        else:
            raise ValueError('Experiment with the same name already exists')

    return ExperimentConfig(
        task=args.task,
        num_epochs=args.num_epochs,
        warmup_epochs=warmup_epochs,
        batch_size=args.batch_size,
        distribution_samples=args.distribution_samples,
        lr=args.lr,
        lr_end=args.lr_end,
        exp_path=exp_path
    )
