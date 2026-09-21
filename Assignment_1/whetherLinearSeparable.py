"""Judge whether a binary-labelled point set is linearly separable."""

import numpy as np
from scipy.optimize import linprog
import matplotlib
import matplotlib.pyplot as plt


def _validate_labelled_data(X):
    """Return X as a checked floating-point array."""
    try:
        data = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a rectangular numeric array.") from exc

    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if data.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if data.shape[1] < 2:
        raise ValueError("Each row of X must contain features and one label.")
    if not np.all(np.isfinite(data)):
        raise ValueError("X cannot contain NaN or infinite values.")

    labels = data[:, -1]
    if not np.all(np.isin(labels, (-1.0, 1.0))):
        raise ValueError("The last column of X must contain only +1 or -1.")

    return data


def _solve_separating_hyperplane(data):
    """Solve for theta = [w_1, ..., w_d, b] when a separator exists."""
    features = data[:, :-1]
    labels = data[:, -1]

    # Include the bias b in theta by appending a constant feature to each row.
    augmented_features = np.column_stack(
        (features, np.ones(features.shape[0], dtype=float))
    )

    # y_i * (w · x_i + b) >= 1 is equivalent to A_ub @ theta <= b_ub.
    A_ub = -(labels[:, np.newaxis] * augmented_features)
    b_ub = -np.ones(data.shape[0], dtype=float)
    objective = np.zeros(augmented_features.shape[1], dtype=float)
    bounds = [(None, None)] * augmented_features.shape[1]

    return linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )


def whetherLinearSeparable(X):
    """Return 1 if the labelled samples are linearly separable, otherwise -1.

    Each row of ``X`` has the form
    ``[feature_1, feature_2, ..., feature_d, label]``, where the label must be
    either +1 or -1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features + 1)
        Samples and their binary labels.

    Returns
    -------
    int
        1 when a separating hyperplane exists; -1 otherwise.
    """
    data = _validate_labelled_data(X)
    result = _solve_separating_hyperplane(data)

    if result.success:
        return 1
    # HiGHS status 2 means that the constraints are infeasible.
    if result.status == 2:
        return -1

    raise RuntimeError("The linear-programming solver failed: " + result.message)


def plotLinearSeparable(X, pdf_path="linear_separable.pdf"):
    """Plot labelled 2D or 3D samples and save the figure as a PDF."""
    data = _validate_labelled_data(X)
    features = data[:, :-1]
    labels = data[:, -1]

    if features.shape[1] not in (2, 3):
        raise ValueError("plotLinearSeparable only supports 2D or 3D feature data.")

    matplotlib.use("Agg", force=True)

    if features.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    positive = labels == 1
    negative = labels == -1

    if features.shape[1] == 2:
        ax.scatter(
            features[negative, 0],
            features[negative, 1],
            c="#d95f02",
            marker="x",
            s=70,
            linewidths=1.8,
            label="label = -1",
        )
        ax.scatter(
            features[positive, 0],
            features[positive, 1],
            c="#1b9e77",
            marker="o",
            s=60,
            edgecolors="black",
            linewidths=0.6,
            label="label = +1",
        )
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
    else:
        ax.scatter(
            features[negative, 0],
            features[negative, 1],
            features[negative, 2],
            c="#d95f02",
            marker="x",
            s=70,
            linewidths=1.8,
            label="label = -1",
        )
        ax.scatter(
            features[positive, 0],
            features[positive, 1],
            features[positive, 2],
            c="#1b9e77",
            marker="o",
            s=60,
            edgecolors="black",
            linewidths=0.6,
            label="label = +1",
        )
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        ax.set_zlabel("feature 3")

    ax.set_title(f"{features.shape[1]}D labelled samples")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)

    return pdf_path


if __name__ == "__main__":
    # Example 1
    Example_1 = np.array(
        [
            [-0.5, 0, -1],
            [3.5, 4.1, -1],
            [4.5, 6, -1],
            [-2, -2.0, -1],
            [-4.1, -2.8, -1],
            [1, 3, -1],
            [-7.1, -4.2, 1],
            [-6.1, -2.2, 1],
            [-4.1, 2.2, 1],
            [1.4, 4.3, 1],
            [-2.4, 4.0, 1],
            [-8.4, -5, 1],
        ]
    )
    # Example 2
    Example_2 = Example_1.copy()
    Example_2[2, -1] = 1
    # Example 3
    Example_3 = np.array([[0, 0, -1], [0, 1, 1], [1, 0, 1], [1, 1, -1]])
    # Example 4
    Example_4 = np.array(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 1],
            [3, 3, 3, 1],
            [-1, -1, -1, -1],
            [-2, -2, -2, -1],
            [-3, -3, -3, -1],
        ]
    )
    # Example 5
    Example_5 = np.array([[0, 0, 1], [0, 0, -1], [1, 1, 1], [1, 1, -1]])
    # Judge whether the labelled samples are linearly separable.
    print("Example 1 (expected 1):", whetherLinearSeparable(Example_1))
    print("Example 2 (expected -1):", whetherLinearSeparable(Example_2))
    print("Example 3 (expected -1):", whetherLinearSeparable(Example_3))
    print("Example 4 (expected 1):", whetherLinearSeparable(Example_4))
    print("Example 5 (expected -1):", whetherLinearSeparable(Example_3))
    # Plot the labelled samples and save the figures as PDF files.
    print("Saved figure:", plotLinearSeparable(Example_1, "Example_1.pdf"))
    print("Saved figure:", plotLinearSeparable(Example_2, "Example_2.pdf"))
    print("Saved figure:", plotLinearSeparable(Example_3, "Example_3.pdf"))
    print("Saved figure:", plotLinearSeparable(Example_4, "Example_4.pdf"))
    print("Saved figure:", plotLinearSeparable(Example_5, "Example_5.pdf"))
