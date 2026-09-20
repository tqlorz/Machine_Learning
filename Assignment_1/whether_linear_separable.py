import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def whetherLinearSeparable(X):
    """
    Determine if a dataset is linearly separable using convex hull method.

    Parameters:
    X : ndarray - Shape (M, N+1) where M is number of samples, N is feature dimension
                  Each row: [feature_1, feature_2, ..., feature_N, label]
                  label is +1 or -1

    Returns:
    Y : int - 1 if linearly separable, -1 if not linearly separable
    """
    X = X.copy().astype(float)
    M, N = X.shape

    # Extract features and labels
    features = X[:, :-1]
    labels = X[:, -1]

    # Separate data by class
    class_pos = features[labels == 1]
    class_neg = features[labels == -1]

    # Check if either class is empty
    if len(class_pos) == 0 or len(class_neg) == 0:
        return 1  # Trivially separable

    # Method 1: Check if convex hulls intersect
    # This is the most geometrically intuitive method
    try:
        result = check_convex_hull_separation(class_pos, class_neg)
        return result
    except:
        # If convex hull method fails (e.g., degenerate cases),
        # fall back to SVM-based method
        pass

    # Method 2: Use linear programming to find separating hyperplane
    try:
        result = check_separability_linear_programming(features, labels)
        return result
    except:
        pass

    # Fallback: return non-separable if both methods fail
    return -1


def check_convex_hull_separation(class_pos, class_neg):
    """
    Check if two point sets are linearly separable by testing
    if their convex hulls are disjoint.
    """
    n_dim = class_pos.shape[1]

    # For 1D or 2D, use direct convex hull intersection test
    if n_dim <= 3:
        return check_hull_intersection_low_dim(class_pos, class_neg)

    # For higher dimensions, use linear programming method
    # to check if convex hulls intersect
    return check_hull_separation_via_lp(class_pos, class_neg)


def check_hull_intersection_low_dim(class_pos, class_neg):
    """
    Check convex hull intersection for low-dimensional cases.
    """
    try:
        # Compute convex hulls
        if len(class_pos) >= class_pos.shape[1] + 1:
            hull_pos = ConvexHull(class_pos)
        else:
            # Too few points for convex hull in this dimension
            hull_pos = None

        if len(class_neg) >= class_neg.shape[1] + 1:
            hull_neg = ConvexHull(class_neg)
        else:
            hull_neg = None

        # Check if any point from one class is inside the other's convex hull
        if hull_pos is not None:
            for point in class_neg:
                if point_in_hull(point, class_pos, hull_pos):
                    return -1  # Hulls intersect, not separable

        if hull_neg is not None:
            for point in class_pos:
                if point_in_hull(point, class_neg, hull_neg):
                    return -1  # Hulls intersect, not separable

        return 1  # Hulls don't intersect, linearly separable

    except:
        # Fall back to LP method
        return check_hull_separation_via_lp(class_pos, class_neg)


def point_in_hull(point, hull_points, hull):
    """
    Check if a point is inside a convex hull.
    """
    try:
        # Add the point to the hull and check if volume increases
        new_points = np.vstack([hull_points, point])
        new_hull = ConvexHull(new_points)
        # If volume doesn't change, point is inside or on the hull
        return np.abs(new_hull.volume - hull.volume) < 1e-10
    except:
        return False


def check_hull_separation_via_lp(class_pos, class_neg):
    """
    Use linear programming to check if convex hulls of two point sets
    can be separated by a hyperplane.

    This solves: can we find w, b such that:
    w^T x_pos + b >= 1 for all x_pos in class_pos
    w^T x_neg + b <= -1 for all x_neg in class_neg
    """
    n_features = class_pos.shape[1]
    n_pos = len(class_pos)
    n_neg = len(class_neg)

    # Variables: [w_1, ..., w_n, b, slack_1, ..., slack_{n_pos+n_neg}]
    n_vars = n_features + 1 + n_pos + n_neg

    # Objective: minimize sum of slack variables
    c = np.zeros(n_vars)
    c[n_features + 1:] = 1  # Penalize slack variables

    # Inequality constraints: -w^T x_pos - b + slack >= -1 (for positive class)
    #                         w^T x_neg + b + slack >= -1 (for negative class)
    A_ub = []
    b_ub = []

    # Positive class constraints: w^T x + b >= 1 - slack
    # Rewrite as: -w^T x - b + slack >= -1
    for i, x in enumerate(class_pos):
        row = np.zeros(n_vars)
        row[:n_features] = -x  # -w^T x
        row[n_features] = -1    # -b
        row[n_features + 1 + i] = 1  # +slack_i
        A_ub.append(row)
        b_ub.append(-1)

    # Negative class constraints: w^T x + b <= -1 + slack
    # Rewrite as: w^T x + b - slack <= -1
    for i, x in enumerate(class_neg):
        row = np.zeros(n_vars)
        row[:n_features] = x    # w^T x
        row[n_features] = 1     # b
        row[n_features + 1 + n_pos + i] = -1  # -slack_i
        A_ub.append(row)
        b_ub.append(-1)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds: slack variables >= 0, w and b unbounded
    bounds = [(None, None)] * (n_features + 1) + [(0, None)] * (n_pos + n_neg)

    # Solve linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        # Check if optimal slack is near zero
        total_slack = np.sum(result.x[n_features + 1:])
        if total_slack < 1e-6:
            return 1  # Linearly separable
        else:
            return -1  # Not linearly separable
    else:
        return -1  # Cannot find separating hyperplane


def check_separability_linear_programming(features, labels):
    """
    Alternative method: directly solve for separating hyperplane.
    """
    n_samples, n_features = features.shape

    # Add bias term
    X_aug = np.column_stack([features, np.ones(n_samples)])

    # Variables: [w_1, ..., w_n, b]
    n_vars = n_features + 1

    # We want: y_i * (w^T x_i + b) >= 1 for all i
    # Rewrite as: -y_i * (w^T x_i + b) <= -1

    A_ub = -labels.reshape(-1, 1) * X_aug
    b_ub = -np.ones(n_samples)

    # Objective: minimize ||w||^2 is non-linear, so just check feasibility
    # Use zero objective (feasibility problem)
    c = np.zeros(n_vars)

    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

    if result.success:
        return 1  # Found separating hyperplane
    else:
        return -1  # No separating hyperplane exists
