# Local data archive

The raw dataset is not committed to this repository. It contains identifiers
that are unnecessary for reviewing the optimizer implementation.

From `L-BFGS/`, download the exact reference archive with:

```bash
python prepare_data.py
```

Or copy and verify an existing archive without network access:

```bash
python prepare_data.py --archive /path/to/data.zip
```

The script writes `data/data.zip`, which Git ignores. It uses verified TLS and
checks the SHA-256 before writing a newly obtained archive. A mismatched existing
file is rejected rather than overwritten.

- Reference repository: https://github.com/Cindy0725/COMP6704-Group-24-Convex-Optimization
- Reference commit: `5624d97b6771cc3814a2063d151db861051976c6`
- Archive SHA-256: `a9c84b961b361a7f41fe80ef85d6b01d90068616e48da7ab134bd2a99e58dcac`
- Dataset provenance: https://www.kaggle.com/datasets/parisrohan/credit-score-classification

The archive must contain `train_data_s.csv`, `train_data_m.csv`,
`train_data_l.csv`, and the labeled `test_data.csv`. These are the reference
project's subsets; downloading Kaggle's full dataset and renaming its files does
not reproduce this experiment.
