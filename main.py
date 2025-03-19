from config import load_config
from experiment import MNIST_Experiment

def main() :
    config = load_config()

    if config.task == 'mnist':
        experiment = MNIST_Experiment(config)
        plot, avg_model = experiment.train()
        plot.plot_metrics()

if __name__ == '__main__':
    main()
