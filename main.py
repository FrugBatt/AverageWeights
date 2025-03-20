from config import load_config
from experiment import MNIST_Experiment

def main() :
    config = load_config()

    if config.task == 'mnist':
        experiment = MNIST_Experiment(config)
    elif config.task == 'mnist_cnn':
        experiment = MNIST_Experiment(config, use_cnn=True)
    else:
        raise ValueError('Task not found')

    plot1, plot2 = experiment.run()

    plot1.save()
    plot2.save()

if __name__ == '__main__':
    main()
