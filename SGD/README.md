# SGD: hinge-loss linear soft-margin SVM

Independent NumPy implementation for COMP6704 Group16. No sklearn estimator/optimizer is used for training.

## Objective

For each one-vs-rest class k, minimize

F_k(w,b) = lambda/2 * ||w||^2 + mean_i max(0, 1 - Y_ik * (x_i^T w + b)).

The bias is not regularized. Mini-batches use their actual sizes, including the final partial batch. At margin exactly1 the hinge subgradient is zero. Each head minimizes its own objective; plots report the mean of the binary objectives. Prediction uses argmax over class scores. Float64, zero initialization, fresh seeded shuffling every epoch, fixed epoch budget and final iterate; no averaging or early stopping.

Learning rate: eta = eta0 / (1 + s/N)^p, where s counts examples processed before an update and N is the current training-set size. p=0,0.5,1 correspond to constant, inverse-square-root and inverse schedules.

## Setup

Recorded runtime: Python 3.11.0. Dependencies are pinned in requirements.txt.

```bash
cd SGD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The 13 tests cover independent numerical examples, finite differences, unregularized bias, hinge boundary behavior, partial-batch normalization, RNG reproducibility, customer-disjoint splitting, preprocessing isolation, and the test-stage lock.

## Reproduce from public source data

Source: [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification). download_data.py uses verified TLS and checks the original labeled train.csv SHA256. A network connection is needed only for downloading; an existing archive can be supplied with --archive.

```bash
python download_data.py --output data/train.csv
python run_experiments.py --data data/train.csv --output runs/results --stage development
python verify_results.py --results runs/results --development-only
python run_experiments.py --output runs/results --stage test
python make_results.py --results runs/results
python verify_results.py --results runs/results
```

Use a fresh empty output directory for a new reproduction. The test stage refuses to run again after test_started.json exists. Do not retune the configuration based on the held-out test results.

## Data and preprocessing

Original labeled source: 100,000 rows from 12,500 customers, eight monthly records/customer. The experiment uniformly samples 1,500 customers with seed 3407, retaining 12,000 rows. Customer-disjoint StratifiedGroupKFold (five folds, seed 2026) gives training/validation/test sizes 7200/2400/2400 and customer counts 900/300/300. This held-out test set comes from the labeled train.csv; Kaggle's unlabeled test.csv is not used.

Class mapping:0=Good,1=Poor,2=Standard. Representation:d=47, K=3. Inputs: 17 numeric features (including parsed credit-history months) and 30 one-hot columns from 4 categorical variables. ID, customer identity, month, name, SSN and free-text Type_of_Loan are excluded as features. Customer ID is used only for sampling/splitting.

Fixed per-row parsing replaces invalid values with missing; ages outside[18,100] and negative numbers except credit-limit changes/payment delays become missing. Train-only preprocessing learns numeric medians,1st/99th percentile clipping bounds, means/stds and categorical one-hot categories. Validation/test use those frozen transforms. Smaller-training-size experiments refit preprocessing only on their own training customers. Data provenance is that of the public Kaggle benchmark, not independently verified real customer outcomes.

## Experiments

All choices are in config.json:

- Pilot:lambda={1e-4,1e-3,1e-2}, eta0={.001,.01,.1,.5}, p={0,.5,1}, batch64,200 epochs, seeds11/22/33. Select by mean validation Macro-F1, then mean accuracy, then deterministic configuration index.
- Learning-rate sensitivity:all12 eta0/p pairs, selected lambda, batch64, evaluation seeds101/202/303/404/505.
- Batch sensitivity:1/16/64/256, selected lambda/eta/p, same200 epochs and five seeds. Equal epochs have unequal update counts and times. Rates are not retuned separately for each batch.
- Stability:all five prespecified evaluation seeds; mean±sample SD (ddof=1), conditional on the fixed customer split.
- Size sensitivity:nested 1800/3600/7200-row training subsets with separately fitted preprocessing, fixed validation customers, five seeds.
- Freeze all development choices and models, then evaluate only the five selected baseline models on test in one final phase. No test sensitivity sweeps.

The protocol generates 193 unique fits after reusing duplicates. Histories distinguish epoch, joint updates, per-head updates, examples processed, optimizer-only time, diagnostic time and total fit wall time. Timers exclude data loading/preprocessing, imports, disk writes, plotting and final test inference. BLAS is limited to one thread. No comparisons to other algorithms are implemented or claimed.

## Source files and generated outputs

- sgd.py:optimizer, dataset preparation and train-only preprocessing.
- run_experiments.py:staged experiment orchestration, artifact hashes and final-test lock.
- download_data.py:download and source-data verification.
- make_results.py:aggregate CSV/LaTeX tables, seven PNG/PDF figure pairs, figure captions and a numerical evidence summary.
- verify_results.py:artifact consistency, split isolation and deterministic training replay; it never predicts test again.
- tests/:correctness and protocol tests.

Generated files go under runs/ (ignored by Git). The source-only repository intentionally omits data snapshots, saved models, predictions, CSV logs and generated results. Environment details and actual numerical results are recorded when the scripts run. Compare cross-algorithm results only after aligning objective, split, preprocessing and timing rules.
