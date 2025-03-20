# AverageWeights

This is a project for the course "Bayesian Machine Learning (MVA)" at ENS Paris-Saclay 

The goal of this project is to implement the "Averaging Weights Leads to Wider Optima and Better Generalization" paper by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson.

With this project, we aim to reproduce the results of the paper on the MNIST and CIFAR dataset. We will also try to extend the results to determine when starting the averaging of the weights is beneficial.

## Authors
- Hugo Fruchet
- Grégoire Dhimoïla

## Usage
You can run 2 types of experiments:
- The first one is to run a training loop on a specific task and verify if the averaging of the weights is beneficial.
- The second one is to run multiple training loops on a specific task and consider the distribution of accuracy in order to determine when starting the averaging of the weights is beneficial in general.

To run the first type of experiment, you can use the following command:
```bash
python main.py --task [task] --num_epochs [num_epochs] --warmup_epochs [warmup_epochs] --batch_size [batch_size] --lr [lr] --experiment_name [experiment_name]
```

with:
- [task]: the task to run the experiment on. It has to be `mnist`, `mnist_cnn`, `cifar10` or `cifar100`)
- [num_epochs]: the total number of epochs to train the model
- [warmup_epochs]: the number of epochs before starting the averaging of the weights. This parameter can be a list to compare different warmup epochs. If it is a list, it has to be separated by commas (e.g. `--warmup_epochs 0,10,20`)
- [batch_size]: the batch size
- [lr]: the learning rate
- [experiment_name]: the name of the experiment (used to save the results)

To run the second type of experiment, you can use the following command:
```bash
python main.py --task [task] --num_epochs [num_epochs] --warmup_epochs [warmup_epochs] --batch_size [batch_size] --distribution_samples [distribution_samples] --lr [lr] --experiment_name [experiment_name]
```

with:
- [task]: the task to run the experiment on. It has to be `mnist`, `mnist_cnn`, `cifar10` or `cifar100`)
- [num_epochs]: the total number of epochs to train the model
- [warmup_epochs]: the number of epochs before starting the averaging of the weights. This parameter can be a list to compare different warmup epochs. If it is a list, it has to be separated by commas (e.g. `--warmup_epochs 0,10,20`)
- [batch_size]: the batch size
- [distribution_samples]: the number of training loops to run in order to determine the distribution of accuracy
- [lr]: the learning rate
- [experiment_name]: the name of the experiment (used to save the results)

Note that the second usage is activated when the `distribution_samples` parameter is set. Otherwise, the first usage is activated.

## Results

The results of the experiments are saved in the `experiments/[experiment_name]` folder. The results are 2 plots saved in a `.png` file.
- In the first usage, you will have a plot of the training metrics (loss and accuracy) and a plot of the accuracy for each warmup epoch.
- In the second usage, you will have a plot of the distribution of accuracy for each warmup epoch and a plot of the increasing of the accuracy for each warmup epoch.

## Requirements
The requirements are listed in the `requirements.txt` file. You can install them with the following command:
```bash
pip install -r requirements.txt
```

## References
- Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson. "Averaging Weights Leads to Wider Optima and Better Generalization". In: arXiv preprint arXiv:1803.05407 (2018).
