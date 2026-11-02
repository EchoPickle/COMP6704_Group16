# Linear Soft-Margin SVM with SMO

An independent NumPy implementation of sequential minimal optimization (SMO)
for a binary linear soft-margin SVM. The optimization package uses NumPy and
the Python standard library. `sklearn.svm.SVC(kernel="linear")` is used only
as an external reference in tests and experiments.

## Objective and scope

The primal objective uses a **sum** of hinge losses:

\[
\min_{w,b}\;\frac12\|w\|^2+C\sum_{i=1}^{n}
\max\{0,1-y_i(w^Tx_i+b)\}.
\]

The bias is unregularized, and labels must contain both `-1` and `+1`.
SMO maximizes the equivalent concave quadratic dual subject to
`0 <= alpha_i <= C` and `y @ alpha == 0`. Two-variable analytical updates
preserve the equality constraint; clipping enforces the box constraints.

For the average-loss convention
`lambda/2 * ||w||^2 + mean(hinge_loss)`, the corresponding coefficient on a
fixed dataset is `C = 1 / (lambda * n)`. This experiment uses synthetic binary
data and its own preprocessing protocol. Its results do not establish a
ranking against the other group implementations unless their objectives,
datasets and evaluation protocols are aligned.

## Setup

Use Python 3.10 or later. From the repository root:

```bash
cd SMO
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install the package and the dependencies for tests and experiments:

```bash
python -m pip install -e ".[test,experiment]"
```

For the optimizer alone, `python -m pip install -e .` installs only NumPy
as a runtime dependency. `requirements.txt` also lists the dependencies
used for testing and experiments.

## Basic use

```python
import numpy as np
from smo_svm import LinearSMO

X = np.array([[-2., -1.], [-1., -1.], [1., 1.], [2., 1.]])
y = np.array([-1, -1, 1, 1])

model = LinearSMO(C=1.0, tol=1e-5, max_iter=2000, random_state=42)
model.fit(X, y)
print(model.decision_function([[0.5, 0.2]]))
print(model.predict([[0.5, 0.2]]))
print(model.converged_, model.max_kkt_violation_, model.duality_gap_)
```

`max_iter` counts scans, not accepted pair updates. A scan can process
several pairs. A fixed `random_state` makes variable selection reproducible;
wall-clock times depend on the machine and software environment.

## Correctness tests

```bash
python -m pytest -q
```

The 82 tests cover feasible pair bounds, analytical updates, duplicate
samples and zero curvature, bias and error-cache updates, dual box and
equality constraints, KKT residuals, primal-dual diagnostics, stopping and
failure states, reproducibility, input validation, train-only scaling and
agreement with linear SVC. A subprocess test blocks `sklearn` and `scipy`
imports while running the solver to check optimization independence.

The reference tests compare objectives and decision functions; they do not
require identical dual multipliers when the dual solution is nonunique.

## Reproduce the representative experiment

```bash
python experiments/run_experiment.py --seed 42 --C 1 --tol 1e-5 --max-iter 3000
```

NumPy generates 300 observations from two overlapping Gaussian classes.
A stratified 70/30 split gives 210 training and 90 test observations.
Both sets are standardized with training-set means and standard deviations.
SMO and SVC receive the same standardized inputs and `C`.

The script writes `results/metrics.json`, `history.csv`, `data.csv`,
`model.npz`, `report.md`, and `experiment.png`, `.pdf`, and `.svg`.
The figure shows the decision boundaries, objective values and maximum
KKT violation. For the reported seed-42 run, SMO took 16 scans and 147
accepted pair updates; test accuracy was 80/90 (88.89%), with 100% test
prediction agreement with SVC. These are results for this synthetic case,
not a general accuracy or speed claim.

## Reproduce the partner-selection study

```bash
python experiments/benchmark_smo.py --sizes 100 300 1000 2000 --seeds 0 1 2 3 4
```

This compares the default heuristic against `RandomPartnerSMO` and linear
SVC. The two SMO policies share the same analytical updates, initialization,
scan schedule, stopping rule and KKT tolerance of `1e-4`; only partner
ordering changes. The heuristic tries free variables first and prioritizes
large error differences. The control shuffles eligible partners.

Each training size has five matched data/optimizer seeds and a separate
1,000-observation test set. Standardization uses training statistics only.
The study limits BLAS threads to one. It saves per-case data and histories,
`runs.csv`, `summary.csv`, `paired_comparisons.csv`, and `study.json` under
`results/study/`. Timing covers one warmed fit per policy and case, including
Gram construction and diagnostics. Variation across seeds is not a
confidence interval. Holding `C` fixed with summed loss also changes the
loss-to-regularization balance as sample size grows.

For a quick check of the commands:

```bash
python experiments/benchmark_smo.py --sizes 100 300 --seeds 0 1
```

## Generate the report and presentation figures

After running both experiments with their defaults:

```bash
python experiments/make_submission_figures.py --output figures
```

This produces convergence, KKT-versus-time, runtime scaling,
partner-selection ablation, decision-boundary and pair-geometry figures
as PDF, SVG and PNG. The scaling plots use medians and the full observed
ranges over the five seeds. The pair-geometry diagram is a mathematical
illustration, not an experimental observation.

Generated results and figures are excluded from Git. The scripts reproduce
them locally; no private data or external student-project results are used.

## Model state and stopping

`fit`, `decision_function` and `predict` are the public methods. Fitted state
includes `alpha_`, `w_`, `bias_`, `support_`, `support_vectors_`, `coef_`,
`intercept_`, `dual_coef_`, `n_iter_`, `n_pair_updates_` and `fit_time_`.
`predict` assigns a zero decision score to `+1`.

`history_` records objective values, primal-dual gap, maximum KKT violation,
number of violations, equality/box errors, scan type, pair-update counts
and elapsed time. `converged_` is true only when the full training set meets
the requested KKT tolerance. Hitting the scan limit or numerical stall emits
`ConvergenceWarning` and retains the current feasible model with a failure
reason. A scan without an update is not automatically treated as convergence.

## Files

```text
src/smo_svm/solver.py                  Pair updates, selection, fitting, prediction
src/smo_svm/diagnostics.py             KKT, objectives and bias interval
src/smo_svm/datasets.py                Synthetic Gaussian generator
tests/                                Component, solver and reference tests
experiments/run_experiment.py          Representative train/test experiment
experiments/benchmark_smo.py           Scaling and partner-selection study
experiments/make_submission_figures.py Report and presentation plots
docs/mathematics.md                    Primal/dual, pair update, bias, KKT derivations
docs/design.md                         Numerical conventions and test mapping
pyproject.toml                        Package metadata and dependencies
requirements.txt                      Test and experiment dependencies
```

Dense finite inputs, binary labels, a linear kernel and a common positive
`C` are supported. Features are not scaled inside the solver. Dense Gram
storage requires `O(n^2)` memory. Sparse inputs, nonlinear kernels,
probabilities, multiclass wrapping and class weights are outside this
implementation's scope.

## Mathematical and reference sources

- J. C. Platt, *Sequential Minimal Optimization: A Fast Algorithm for Training
  Support Vector Machines*, Microsoft Research Technical Report MSR-TR-98-14,
  1998: <https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf>.
- scikit-learn SVC API documentation:
  <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>.
