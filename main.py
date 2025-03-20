from config import load_config
from experiment import MNIST_Experiment, CIFAR_Experiment

def main() :
    config = load_config()

    if config.task == 'mnist':
        experiment = MNIST_Experiment(config)
    elif config.task == 'mnist_cnn':
        experiment = MNIST_Experiment(config, use_cnn=True)
    elif config.task == 'cifar10':
        experiment = CIFAR_Experiment(config)
    elif config.task == 'cifar100':
        experiment = CIFAR_Experiment(config, cifar100=True)
    else:
        raise ValueError('Task not found')

    plot1, plot2 = experiment.run()

    plot1.save()
    plot2.save()

if __name__ == '__main__':
    main()
