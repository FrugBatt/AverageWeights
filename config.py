import argparse

class ExperimentConfig:

    def __init__(self,
        task,
        num_epochs,
        warmup_epochs,
        batch_size,
        lr,
        lr_end
    ):
        self.task = task
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_end = lr_end

def load_config():
    parser = argparse.ArgumentParser(description='Average Weights Experiments')
    parser.add_argument('--task', type=str, default='mnist', help='Task to run')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs (for the fine-tuned average)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (initial learning rate if scheduler is used)')
    parser.add_argument('--lr_end', type=float, default=None, help='Enable the learning rate linear scheduler and final learning rate')

    args = parser.parse_args()

    return ExperimentConfig(
        task=args.task,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_end=args.lr_end
    )
