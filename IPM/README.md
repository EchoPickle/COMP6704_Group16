# IPM: kernel soft-margin SVM via a log-barrier interior point method

Independent NumPy implementation for COMP6704 Group16. The Newton system is solved with
`numpy.linalg.solve`; no external quadratic-programming solver is used for training. scikit-learn
is used only for preprocessing (`StandardScaler`, `LabelEncoder`) and scoring (`accuracy_score`,
`classification_report`).

## Objective

Training uses the dual of the kernel soft-margin SVM in the Lagrange multipliers
`alpha in R^n`:

```
maximize  sum_i alpha_i - 1/2 * sum_i sum_j alpha_i alpha_j y_i y_j K_ij
subject to  0 <= alpha_i <= C
```

The box constraint is handled by a primal logarithmic barrier rather than by projection, so
every iterate stays strictly inside `(0, C)`. For a barrier parameter `mu > 0` we minimize

```
f_mu(alpha) = f(alpha) - mu * sum_i [ log(alpha_i) + log(C - alpha_i) ],
f(alpha)    = 1/2 * alpha^T K alpha - 1^T alpha
```

with gradient `g = 1 - K alpha + mu * (-1/alpha + 1/(C - alpha))` and Hessian
`H = K + mu * diag(1/alpha_i^2 + 1/(C - alpha_i)^2)`, all element-wise. Each iteration takes the
Newton direction `d = -H^{-1} g`, damps it with a backtracking line search (halve the step from
`t = 1` until `alpha + t d` lies in `(0, C)`), clips to `[1e-10, C - 1e-10]`, and then reduces the
barrier geometrically with `mu <- 0.9 * mu`. Initial values are `alpha = 0.1 * ones(n)` and
`mu = 1.0`. The loop stops when `||d||_2 < 1e-6` or the iteration budget is reached.

Free support vectors are identified by the band `1e-6 < alpha_i < C - 1e-6`; the intercept is the
mean of `y_i - sum_j alpha_j y_j K_ij` over that band and is set to zero when the band is empty.
Multiclass problems use a one-vs-one decomposition with majority voting, so three classes require
three binary solves per experiment.

Kernels: `linear` (`x^T x'`), `rbf` (`exp(-gamma * ||x - x'||^2)`, default `gamma = 0.1`),
`poly` (`(gamma * x^T x' + coef0)^degree`, default `degree = 3`, `coef0 = 1.0`) and `sigmoid`.
Default box bound `C = 1.0`.

## Setup

Recorded runtime: Python 3.8. Dependencies are listed in requirements.txt.

```bash
cd IPM
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Data

Data is not committed. The scripts expect a `data/` directory next to them:

```
data/train_data_s.csv
data/train_data_m.csv
data/train_data_l.csv
data/test_data.csv
```

Source: [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).
The splits used here are the project's shared splits: training sizes 4,782 (S), 7,932 (M) and
11,122 (L), with one common test set of 6,939 rows. Keep these splits and the SVM objective
aligned when comparing across algorithms.

## Preprocessing

Parsing is applied per row before scaling: `Credit_History_Age` ("5 Years and 3 Months" style) is
converted to a fractional number of years, `Credit_Mix` is mapped to `{Good: 1.0, Standard: 0.5,
Bad: 0.0}`, and `Payment_Behaviour` is label-encoded from categories observed in the training set.
Rows containing missing values are then dropped. Eighteen numerical features are kept
(`Age`, `Annual_Income`, `Monthly_Inhand_Salary`, `Num_Bank_Accounts`, `Num_Credit_Card`,
`Interest_Rate`, `Num_of_Loan`, `Delay_from_due_date`, `Num_of_Delayed_Payment`,
`Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Mix`, `Outstanding_Debt`,
`Credit_Utilization_Ratio`, `Credit_History_Age`, `Amount_invested_monthly`, `Monthly_Balance`,
`Payment_Behaviour`). Labels are label-encoded and every feature is standardised, with the scaler
fitted on the training rows only.

## Reproduce

Quick subsampled run (small enough for a laptop, but its accuracy and timings are **not** the ones
reported below):

```bash
python run_ipm_local.py --datascale s --kernel linear --data-folder data --output-dir outputs/ipm
python make_ipm_experiment_figure.py
```

`run_ipm_local.py` writes `outputs/ipm/result_<scale>_<kernel>.json` and a convergence plot
`outputs/ipm/all_classifiers_convergence_<kernel>_<scale>.png`. Defaults:
`--sample-per-class 30 --test-per-class 120 --max-iter 25 --C 1.0 --gamma 0.1 --degree 3
--coef0 1.0`.

The full-scale runs reported below were produced by `COMP6704_Group24_IPM.ipynb` (Colab; set
`data_folder` to the directory holding the split CSVs). That notebook loops over
`datascale in {s, m, l}` and `kernel in {linear, rbf, poly}`, solves each problem without
subsampling, and appends one row per run to `result_data.csv`.

## Experiments and results

Nine full-scale runs: three nested training sizes x three kernels, evaluated on the shared
6,939-row test set. Majority-class baseline: 0.5468.

| Scale      | Linear            | RBF               | Polynomial        |
|------------|-------------------|-------------------|-------------------|
| S (4,782)  | 0.5793 / 446 s    | 0.5469 / 452 s    | 0.5687 / 394 s    |
| M (7,932)  | 0.5896 / 1,795 s  | 0.5473 / 1,823 s  | 0.5763 / 1,778 s  |
| L (11,122) | 0.5829 / 4,517 s  | 0.5485 / 4,584 s  | 0.5756 / 4,562 s  |

Each cell is accuracy / total training time in seconds. Accuracy is nearly flat across scales
while runtime grows about tenfold from S to L, consistent with the dense `n x n` solve performed
once per Newton iteration. The linear kernel gives the only consistent margin over the baseline;
the RBF kernel at `gamma = 0.1` stays within 0.2 percentage points of it.

Per-classifier timings from the notebook show that the one-vs-one subproblems are unbalanced: in
the S-scale linear run they take 51.9 s (Good vs Poor, 2,167 rows), 168.6 s (Good vs Standard,
3,492 rows) and 223.9 s (Poor vs Standard, 3,905 rows), so the slowest pair sets the wall-clock
time of the whole experiment.

## Known limitations

- The equality constraint `sum_i alpha_i y_i = 0` is not enforced, so this is a teaching-oriented
  kernel SVM solver rather than a full textbook dual solver.
- `run_ipm_local.py` subsamples the splits (30 training rows per class by default) to keep the
  dense solve tractable; only the notebook reproduces the full-scale numbers in the table above.
- Results depend on the fixed `gamma`, `degree` and `coef0` values; no kernel hyperparameter
  search is performed, so the comparison is about optimisation behaviour rather than about
  reaching state-of-the-art accuracy.

## Source files

- `run_ipm_local.py`: solver (kernels, log-barrier Newton loop, one-vs-one wrapper, preprocessing)
  and a command-line runner for a single scale/kernel experiment.
- `COMP6704_Group24_IPM.ipynb`: full-scale experiment driver and the recorded run log.
- `make_ipm_experiment_figure.py`: two-panel accuracy/runtime summary figure from the recorded
  numbers in the table above; writes `outputs/ipm/ipm_experiment_summary.png`.

Generated files go under `outputs/` and are not committed.
