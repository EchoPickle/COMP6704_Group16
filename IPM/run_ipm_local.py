import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


credit_mix_mapping = {
    "Good": 1.0,
    "Standard": 0.5,
    "Bad": 0.0,
}


def convert_to_float(age_str):
    if isinstance(age_str, str):
        years, months = map(
            int, age_str.replace("Years and", "").replace("Months", "").split()
        )
        return years + months / 12.0
    return age_str


def compute_kernel(X1, X2, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
    if kernel == "linear":
        return X1 @ X2.T
    if kernel == "rbf":
        sq_dists = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1)
            - 2 * X1 @ X2.T
        )
        return np.exp(-gamma * sq_dists)
    if kernel == "poly":
        return (gamma * X1 @ X2.T + coef0) ** degree
    if kernel == "sigmoid":
        return np.tanh(gamma * X1 @ X2.T + coef0)
    raise ValueError(f"Unsupported kernel: {kernel}")


def interior_point_svm_binary(
    X,
    y,
    C=1.0,
    kernel="rbf",
    gamma=1.0,
    degree=3,
    coef0=1.0,
    tol=1e-6,
    max_iter=100,
    collect_data=False,
):
    start_time = time.time()
    n, _ = X.shape
    K = compute_kernel(X, X, kernel, gamma, degree, coef0)
    alpha = np.ones(n) * 0.1
    mu = 1.0

    residuals, objectives, mus = [], [], []

    for iteration in range(max_iter):
        barrier_hess = 1 / (alpha**2) + 1 / ((C - alpha) ** 2)
        H = K + mu * np.diag(barrier_hess)
        barrier_grad = -1 / alpha + 1 / (C - alpha)
        grad = np.ones(n) - K @ alpha + mu * barrier_grad
        delta_alpha = np.linalg.solve(H, -grad)

        step_size = 1.0
        while np.any(alpha + step_size * delta_alpha <= 0) or np.any(
            alpha + step_size * delta_alpha >= C
        ):
            step_size *= 0.5
            if step_size < 1e-10:
                break

        alpha += step_size * delta_alpha
        alpha = np.clip(alpha, 1e-10, C - 1e-10)

        residual = np.linalg.norm(delta_alpha)
        objective = 0.5 * np.sum(alpha * (K @ alpha)) - np.sum(alpha)

        residuals.append(float(residual))
        objectives.append(float(objective))
        mus.append(float(mu))

        mu *= 0.9

        if residual < tol:
            print(f"  stop iteration: {iteration}")
            break

    support_idx = (alpha > 1e-6) & (alpha < C - 1e-6)
    if kernel == "linear":
        w = X.T @ (alpha * y)
        b = np.mean(y[support_idx] - X[support_idx] @ w) if np.any(support_idx) else 0.0
        training_time = time.time() - start_time
        if collect_data:
            return (w, b), training_time, (residuals, objectives, mus)
        return (w, b), training_time

    if np.any(support_idx):
        K_sv = compute_kernel(X[support_idx], X, kernel, gamma, degree, coef0)
        b = np.mean(y[support_idx] - np.sum((alpha * y) * K_sv, axis=1))
    else:
        b = 0.0

    model = {
        "alpha": alpha,
        "X": X,
        "y": y,
        "b": float(b),
        "kernel": kernel,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
    }

    training_time = time.time() - start_time
    if collect_data:
        return model, training_time, (residuals, objectives, mus)
    return model, training_time


def predict_nonlinear(X_test, model):
    K_test = compute_kernel(
        X_test,
        model["X"],
        model["kernel"],
        model["gamma"],
        model["degree"],
        model["coef0"],
    )
    return np.sign(np.sum((model["alpha"] * model["y"]) * K_test, axis=1) + model["b"])


def plot_convergence(all_residuals, labels, output_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    for (residuals, objectives, mus), label in zip(all_residuals, labels):
        ax1.semilogy(residuals, label=label)
        ax2.plot(objectives, label=label)
        ax3.semilogy(mus, label=label)

    ax1.set_title("Residual Convergence")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual (log scale)")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Objective Function")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Objective Value")
    ax2.legend()
    ax2.grid(True)

    ax3.set_title("Barrier Parameter (Mu)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Mu (log scale)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def multiclass_svm_ovo(
    X_train,
    y_train,
    X_test,
    classes,
    output_dir,
    datascale="",
    C=1.0,
    kernel="rbf",
    gamma=1.0,
    degree=3,
    coef0=1.0,
    max_iter=100,
):
    n_classes = len(classes)
    classifiers = {}
    all_residuals = []
    labels = []

    print(f"Training {n_classes * (n_classes - 1) // 2} binary classifiers...")

    classifier_count = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            classifier_count += 1
            print(f"\nTraining classifier {classifier_count}: classes {classes[i]} vs {classes[j]}")

            mask = (y_train == classes[i]) | (y_train == classes[j])
            X_pair = X_train[mask]
            y_pair = y_train[mask]
            y_binary = np.where(y_pair == classes[i], 1, -1)
            print(f"  pair sample size: {len(X_pair)}")

            classifier, train_time, convergence = interior_point_svm_binary(
                X_pair,
                y_binary,
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                max_iter=max_iter,
                collect_data=True,
            )
            print(f"  training time: {train_time:.4f} seconds")
            classifiers[(i, j)] = classifier
            all_residuals.append(convergence)
            labels.append(f"Class {classes[i]} vs {classes[j]}")

    plot_path = output_dir / f"all_classifiers_convergence_{kernel}_{datascale}.png"
    plot_convergence(all_residuals, labels, plot_path)

    predictions = []
    for x in X_test:
        votes = np.zeros(n_classes)
        for (i, j), classifier in classifiers.items():
            if kernel == "linear":
                w, b = classifier
                pred = np.sign(x @ w + b)
            else:
                pred = predict_nonlinear(x.reshape(1, -1), classifier)[0]
            if pred > 0:
                votes[i] += 1
            else:
                votes[j] += 1
        predictions.append(classes[np.argmax(votes)])

    return np.array(predictions), plot_path


def processing_train_test(train_data, test_data):
    train_data["Credit_History_Age"] = train_data["Credit_History_Age"].apply(convert_to_float)
    test_data["Credit_History_Age"] = test_data["Credit_History_Age"].apply(convert_to_float)
    train_data["Credit_Mix"] = train_data["Credit_Mix"].map(credit_mix_mapping)
    test_data["Credit_Mix"] = test_data["Credit_Mix"].map(credit_mix_mapping)

    unique_behaviors = train_data["Payment_Behaviour"].unique()
    behavior_mapping = {behavior: idx for idx, behavior in enumerate(unique_behaviors)}
    train_data["Payment_Behaviour"] = train_data["Payment_Behaviour"].map(behavior_mapping)
    test_data["Payment_Behaviour"] = test_data["Payment_Behaviour"].map(behavior_mapping)

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    features = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Age",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Payment_Behaviour",
    ]

    X_train = train_data[features]
    y_train = train_data["Credit_Score"]
    X_test = test_data[features]
    y_test = test_data["Credit_Score"]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, le


def stratified_cap(X, y, max_per_class=None, random_state=42):
    if max_per_class is None:
        return X, y

    rng = np.random.default_rng(random_state)
    keep_indices = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        keep_indices.append(np.sort(cls_idx))

    keep_indices = np.sort(np.concatenate(keep_indices))
    return X[keep_indices], y[keep_indices]


def run_svm_experiment(
    datascale="s",
    kernel="rbf",
    data_folder="data",
    output_dir="outputs/ipm",
    C=1.0,
    gamma=0.1,
    degree=3,
    coef0=1.0,
    sample_per_class=30,
    test_per_class=120,
    max_iter=25,
    random_state=42,
):
    data_folder = Path(data_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = data_folder / f"train_data_{datascale}.csv"
    test_data_path = data_folder / "test_data.csv"

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    print(f"\nExperiment: datascale={datascale}, kernel={kernel}")
    X_train, y_train, X_test, y_test, label_encoder = processing_train_test(train_data, test_data)
    X_train, y_train = stratified_cap(X_train, y_train, sample_per_class, random_state)
    X_test, y_test = stratified_cap(X_test, y_test, test_per_class, random_state + 1)
    classes = np.unique(y_train)

    print(f"Train size after preprocessing/sampling: {len(X_train)}")
    print(f"Test size after preprocessing/sampling: {len(X_test)}")
    print("Train class counts:", {int(cls): int(np.sum(y_train == cls)) for cls in classes})

    total_start_time = time.time()
    y_pred, plot_path = multiclass_svm_ovo(
        X_train,
        y_train,
        X_test,
        classes,
        output_dir=output_dir,
        datascale=datascale,
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        max_iter=max_iter,
    )
    total_time = time.time() - total_start_time

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.inverse_transform(classes),
        zero_division=0,
    )

    result = {
        "datascale": datascale,
        "kernel": kernel,
        "timing": total_time,
        "accuracy": float(accuracy),
        "sample_per_class": sample_per_class,
        "test_per_class": test_per_class,
        "max_iter": max_iter,
        "plot_path": str(plot_path),
    }

    result_path = output_dir / f"result_{datascale}_{kernel}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nAccuracy: {accuracy:.4f}, Time: {total_time:.4f}s")
    print("\nClassification report:")
    print(report)
    print(f"Saved convergence plot to: {plot_path}")
    print(f"Saved result summary to: {result_path}")

    return result, report


def main():
    parser = argparse.ArgumentParser(description="Run the local IPM SVM experiment.")
    parser.add_argument("--datascale", default="s", choices=["s", "m", "l"])
    parser.add_argument("--kernel", default="linear", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--data-folder", default="data")
    parser.add_argument("--output-dir", default="outputs/ipm")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--coef0", type=float, default=1.0)
    parser.add_argument("--sample-per-class", type=int, default=30)
    parser.add_argument("--test-per-class", type=int, default=120)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_svm_experiment(
        datascale=args.datascale,
        kernel=args.kernel,
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        C=args.C,
        gamma=args.gamma,
        degree=args.degree,
        coef0=args.coef0,
        sample_per_class=args.sample_per_class,
        test_per_class=args.test_per_class,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
