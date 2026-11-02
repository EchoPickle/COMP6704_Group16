# COMP6704 Group 16

Group project: training support vector machines with convex optimization algorithms.

Each algorithm is kept in its own directory so members can contribute independently.

## SGD

[SGD/](SGD/) contains an independent NumPy implementation of mini-batch stochastic subgradient descent for the hinge-loss soft-margin linear SVM, correctness tests, hyperparameter configuration, and scripts for reproducing experiments and generating tables/figures.

See [SGD/README.md](SGD/README.md) for setup and reproduction commands.

This initial upload contains source code and configuration only. Datasets, saved models, predictions, logs and generated experiment results are not included; they can be generated locally using the scripts.
## IPM

[IPM/](IPM/) contains an independent NumPy implementation of a primal logarithmic-barrier interior point method for the kernel soft-margin SVM, covering the log-barrier Newton solver, a one-vs-one multiclass wrapper, the data preprocessing pipeline, and a script for reproducing the experiments and generating the summary figure.

See [IPM/README.md](IPM/README.md) for setup and reproduction commands.

This initial upload contains source code and configuration only. Datasets, saved models, predictions, logs and generated experiment results are not included; they can be generated locally using the scripts.

## SMO

[SMO/](SMO/) contains an independent NumPy implementation of analytical
sequential minimal optimization for the binary linear soft-margin SVM,
82 correctness tests, mathematical derivations, an SVC reference comparison,
a paired partner-selection study, and scripts for reproducing experiments
and generating report/presentation figures.

See [SMO/README.md](SMO/README.md) for installation, the summed-hinge-loss
objective, data/preprocessing details, and reproduction commands. This
contribution includes source code and configuration; synthetic data, models,
logs and figures are generated locally by the scripts.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for uploading an algorithm implementation. Each member should keep their implementation in its own directory and submit changes through a branch/pull request.
