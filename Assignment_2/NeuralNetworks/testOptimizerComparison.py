import argparse
import copy
import os
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NN import NN
from nn_test import nn_test
from nn_train import nn_train


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "krkopt.data")
NETWORK_LAYERS = [6, 20, 20, 20, 20, 2]

EXPERIMENTS = (
    ("Momentum", 0.01),
    ("RMSPropNesterov", 0.001),
)


def load_chess_data():
    with open(DATA_PATH, encoding="utf-8") as data_file:
        # Keep the same preprocessing behavior as testChess.py.
        rows = data_file.readlines()[1:]

    data = np.zeros((len(rows), 6), dtype=float)
    labels = np.zeros((len(rows), 2), dtype=float)

    for index, row in enumerate(rows):
        fields = row.strip().split(",")
        data[index] = [
            ord(fields[0]) - 96,
            int(fields[1]),
            ord(fields[2]) - 96,
            int(fields[3]),
            ord(fields[4]) - 96,
            int(fields[5]),
        ]
        labels[index] = [1, 0] if fields[6] == "draw" else [0, 1]

    x_train, x_remaining, y_train, y_remaining = train_test_split(
        data,
        labels,
        test_size=0.6,
        random_state=1,
    )
    x_test, x_validation, y_test, y_validation = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.2,
        random_state=1,
    )

    scaler = StandardScaler(copy=False)
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_validation)
    scaler.transform(x_test)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def create_network(method, learning_rate, seed, batch_size):
    np.random.seed(seed)
    return NN(
        layer=NETWORK_LAYERS,
        active_function="sigmoid",
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimization_method=method,
        batch_normalization=1,
        objective_function="Cross Entropy",
    )


def snapshot_parameters(nn):
    return {
        "W": {k: value.copy() for k, value in nn.W.items()},
        "b": {k: value.copy() for k, value in nn.b.items()},
        "Gamma": {k: np.array(value, copy=True) for k, value in nn.Gamma.items()},
        "Beta": {k: np.array(value, copy=True) for k, value in nn.Beta.items()},
    }


def assert_same_initial_parameters(left, right):
    for parameter_name in ("W", "b", "Gamma", "Beta"):
        left_values = left[parameter_name]
        right_values = right[parameter_name]
        if left_values.keys() != right_values.keys():
            raise AssertionError(f"{parameter_name} layer indices are different")

        for layer in left_values:
            if not np.array_equal(left_values[layer], right_values[layer]):
                raise AssertionError(
                    f"Initial {parameter_name}[{layer}] values are different"
                )


def assert_zero_initial_velocity(nn):
    for group in (nn.vW, nn.vb, nn.vGamma, nn.vBeta):
        for value in group.values():
            if np.any(np.asarray(value) != 0):
                raise AssertionError("Optimizer velocity must be initialized to zero")


def build_confusion_matrix(y_true, y_pred):
    matrix = np.zeros((2, 2), dtype=int)
    true_labels = np.argmax(y_true, axis=1)
    for true_label, predicted_label in zip(true_labels, y_pred):
        matrix[true_label, predicted_label] += 1
    return matrix


def train_experiment(
    method,
    learning_rate,
    seed,
    epochs,
    batch_size,
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    save_model,
):
    nn = create_network(method, learning_rate, seed, batch_size)
    initial_parameters = snapshot_parameters(nn)
    assert_zero_initial_velocity(nn)

    best_accuracy = -1.0
    best_epoch = 0
    best_nn = None
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Use the same mini-batch permutation for both optimizers.
        np.random.seed(seed + epoch)
        cost_count_before_epoch = len(nn.cost)
        nn = nn_train(nn, x_train, y_train)

        new_costs = list(nn.cost.values())[cost_count_before_epoch:]
        epoch_cost = float(np.mean(new_costs))
        _, _, validation_accuracy, _ = nn_test(nn, x_validation, y_validation)
        validation_accuracy = float(validation_accuracy)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_nn = copy.deepcopy(nn)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{method}] epoch={epoch:3d} "
                f"cost={epoch_cost:.6f} "
                f"validation_accuracy={validation_accuracy:.6f}"
            )

    elapsed_seconds = time.perf_counter() - started_at
    wrongs, predictions, test_accuracy, _ = nn_test(best_nn, x_test, y_test)
    matrix = build_confusion_matrix(y_test, predictions)

    model_path = os.path.join(BASE_PATH, f"storedChess_{method}.pkl")
    if save_model:
        with open(model_path, "wb") as model_file:
            pickle.dump(best_nn, model_file)

    return {
        "method": method,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "test_accuracy": float(test_accuracy),
        "test_errors": int(np.sum(wrongs)),
        "confusion_matrix": matrix,
        "elapsed_seconds": elapsed_seconds,
        "model_path": model_path if save_model else None,
        "initial_parameters": initial_parameters,
    }


def print_summary(results):
    print("\nComparison")
    print(
        "method              lr       best_epoch  validation  test       errors  "
        "seconds"
    )
    for result in results:
        print(
            f"{result['method']:<19} "
            f"{result['learning_rate']:<8g} "
            f"{result['best_epoch']:<11d} "
            f"{result['validation_accuracy']:<11.6f} "
            f"{result['test_accuracy']:<10.6f} "
            f"{result['test_errors']:<7d} "
            f"{result['elapsed_seconds']:.3f}"
        )
        print(f"confusion_matrix ({result['method']}):")
        print(result["confusion_matrix"])
        if result["model_path"]:
            print(f"saved_model: {result['model_path']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Momentum and RMSPropNesterov from identical initial models "
            "and with identical mini-batch permutations."
        )
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the best model from each experiment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = load_chess_data()
    x_train, _, x_validation, _, x_test, _ = datasets
    print(
        f"samples: train={len(x_train)}, validation={len(x_validation)}, "
        f"test={len(x_test)}"
    )
    print(f"seed={args.seed}, epochs={args.epochs}, batch_size={args.batch_size}")

    results = []
    for method, learning_rate in EXPERIMENTS:
        results.append(
            train_experiment(
                method,
                learning_rate,
                args.seed,
                args.epochs,
                args.batch_size,
                *datasets,
                save_model=not args.no_save,
            )
        )

    assert_same_initial_parameters(
        results[0]["initial_parameters"],
        results[1]["initial_parameters"],
    )
    print("initial model check: passed")
    print_summary(results)


if __name__ == "__main__":
    main()
