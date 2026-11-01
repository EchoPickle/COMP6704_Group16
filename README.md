# COMP6704 Group 16

Group project: training support vector machines with convex optimization algorithms.

Each algorithm is kept in its own directory so members can contribute independently.

## SGD

[SGD/](SGD/) contains an independent NumPy implementation of mini-batch stochastic subgradient descent for the hinge-loss soft-margin linear SVM, correctness tests, hyperparameter configuration, and scripts for reproducing experiments and generating tables/figures.

See [SGD/README.md](SGD/README.md) for setup and reproduction commands.

This initial upload contains source code and configuration only. Datasets, saved models, predictions, logs and generated experiment results are not included; they can be generated locally using the scripts.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for uploading an algorithm implementation. Each member should keep their implementation in its own directory and submit changes through a branch/pull request.
