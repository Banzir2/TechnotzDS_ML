import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ==========================================
# 0. Preprocessing
# ==========================================


def show_random_samples_fixed(images, labels, height=57, width=72, num=10):
    plt.figure(figsize=(18, 4))
    indices = random.sample(range(len(images)), min(num, len(images)))

    for i, idx in enumerate(indices):
        img = images[idx]

        # Check if the pixel count matches your expected reshape
        if img.size != height * width:
            print(
                f"Error: Image size is {img.size}, but you are trying to reshape to {height}x{width}"
            )
            # Dynamic fix attempt (assuming height remains 72)
            width = int(img.size / height)

        plt.subplot(1, num, i + 1)
        plt.imshow(img.reshape(height, width), cmap="gray")
        plt.title(f"{labels[idx]}")
        plt.axis("off")
    plt.show()


def crop_grey_strips(img, strip_width=7, target_color=127, tolerance=10):
    """
    Detects if a 127-intensity (7F7F7F) grey strip exists on the left or right
    and crops it. Returns a resized image to maintain consistency.
    """
    h, w = img.shape

    # Calculate the average intensity of the far left and far right columns
    left_side_avg = np.mean(img[:, :strip_width])
    right_side_avg = np.mean(img[:, -strip_width:])

    # Determine crop boundaries
    start_col = 0
    end_col = w

    # Check if left side matches the grey strip color
    if abs(left_side_avg - target_color) <= tolerance:
        start_col = strip_width

    # Check if right side matches (using 'elif' assumes only one side has it)
    elif abs(right_side_avg - target_color) <= tolerance:
        end_col = w - strip_width

    # Perform the crop
    cropped_img = img[:, start_col:end_col]

    # CRITICAL: Resize back to the original width (64) or a new fixed width (55)
    # If we don't resize, the flattened vectors will have different lengths.
    # Let's resize to 72x55 since that's the "real" face data area.
    final_img = cv2.resize(cropped_img, (72, 57), interpolation=cv2.INTER_AREA)

    return final_img


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_img = clahe.apply(img)
    target_mean = 127
    current_mean = cv2.mean(final_img)[0]
    rescaled_img = cv2.convertScaleAbs(
        final_img, alpha=target_mean / current_mean, beta=0
    )
    rescaled_img = 255 - rescaled_img
    return rescaled_img


def load_filtered_data(directory, threshold=20):
    images = []
    labels = []
    removed_count = 0

    path_list = list(Path(directory).glob("*.pgm"))

    for img_path in path_list:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = crop_grey_strips(img)
        img = apply_clahe(img)

        if img is not None:
            # Calculate the average brightness of the image
            if np.mean(img) > threshold:
                images.append(img.flatten())
                labels.append(img_path.name.split("_")[0].removeprefix("p"))
            else:
                removed_count += 1

    print(f"Directory: {directory}")
    print(f"Kept: {len(images)} | Removed (too dark): {removed_count}")
    return np.array(images), np.array(labels)


# Example usage:
X_train, y_train = load_filtered_data("Train Set (Split)/train", threshold=0)
X_test, y_test = load_filtered_data("Train Set (Split)/test", threshold=0)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# --- 0. ENCODE LABELS FOR PLOTTING ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# --- 1. RESHAPE RAW DATA FOR HOG ---
X_train_images = X_train.reshape(-1, 57, 72)
X_test_images = X_test.reshape(-1, 57, 72)


# --- 2. EXTRACT HOG FEATURES ---
def get_hog_features(images):
    features_list = []
    for img in images:
        fd = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
        )
        features_list.append(fd)
    return np.array(features_list)


print("Extracting HOG features... (this might take a few seconds)")
X_train_hog = get_hog_features(X_train_images)
X_test_hog = get_hog_features(X_test_images)

# --- 3. SCALE BOTH DATASETS INDEPENDENTLY ---
scaler_raw = StandardScaler()
X_train_raw_scaled = scaler_raw.fit_transform(X_train)
X_test_raw_scaled = scaler_raw.transform(X_test)

scaler_hog = StandardScaler()
X_train_hog_scaled = scaler_hog.fit_transform(X_train_hog)
X_test_hog_scaled = scaler_hog.transform(X_test_hog)

# --- 4. APPLY PCA TO COMPRESS BOTH ---
pca_hog = PCA(n_components=150, whiten=True)
X_train_hog_pca = pca_hog.fit_transform(X_train_hog_scaled)
X_test_hog_pca = pca_hog.transform(X_test_hog_scaled)

pca_raw = PCA(n_components=150, whiten=True)
X_train_raw_pca = pca_raw.fit_transform(X_train_raw_scaled)
X_test_raw_pca = pca_raw.transform(X_test_raw_scaled)

# --- 5. CONCATENATE (GLUE) THEM TOGETHER ---
X_train_final = np.hstack((X_train_hog_pca, X_train_raw_pca))
X_test_final = np.hstack((X_test_hog_pca, X_test_raw_pca))

np.savez_compressed("data/X_train_final.npz", data=X_train_final)
np.savez_compressed("data/y_train.npz", data=y_train)
np.savez_compressed("data/X_test_final.npz", data=X_test_final)
np.savez_compressed("data/y_test.npz", data=y_test)

print(f"Final training data shape: {X_train_final.shape}")

# ==========================================
# 1. Custom Model Classes (Adapted for Sklearn)
# ==========================================


class MultinomialLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.1, epochs=1000, reg_type=None, lambda_param=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg_type = reg_type
        self.lambda_param = lambda_param
        self.W = None
        self.b = None

    def _one_hot(self, y, num_classes):
        out = np.zeros((y.size, num_classes))
        out[np.arange(y.size), y.astype(int)] = 1
        return out

    def _softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        y_onehot = self._one_hot(y, num_classes)

        self.W = np.random.randn(num_features, num_classes) * 0.01
        self.b = np.zeros(num_classes)

        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            probs = self._softmax(logits)
            error = probs - y_onehot

            dW = (1 / num_samples) * (X.T @ error)
            db = (1 / num_samples) * np.sum(error, axis=0)

            if self.reg_type == "l2":
                dW += self.lambda_param * self.W
            elif self.reg_type == "l1":
                dW += self.lambda_param * np.sign(self.W)

            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict(self, X):
        logits = X @ self.W + self.b
        return np.argmax(self._softmax(logits), axis=1)


# --- Tree Classes (Node & DecisionTree needed for Forest) ---


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature, self.threshold, self.left, self.right, self.value = (
            feature,
            threshold,
            left,
            right,
            value,
        )

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_features = int(np.sqrt(X.shape[1]))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_labels = X.shape[0], len(np.unique(y))
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            return Node(value=Counter(y).most_common(1)[0][0])

        feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()
        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1),
            right=self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1),
        )

    def _best_split(self, X, y, feat_idxs):
        best_gain, split_idx, split_thresh = -1, None, None
        for feat_idx in feat_idxs:
            thresholds = np.percentile(X[:, feat_idx], [25, 50, 75])
            for thr in thresholds:
                gain = self._info_gain(y, X[:, feat_idx], thr)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thr
        return split_idx, split_thresh

    def _info_gain(self, y, X_col, thr):
        p_imp = self._gini(y)
        l_idx, r_idx = (
            np.argwhere(X_col <= thr).flatten(),
            np.argwhere(X_col > thr).flatten(),
        )
        if len(l_idx) == 0 or len(r_idx) == 0:
            return 0
        return (
            p_imp
            - (len(l_idx) / len(y)) * self._gini(y[l_idx])
            - (len(r_idx) / len(y)) * self._gini(y[r_idx])
        )

    def _gini(self, y):
        probs = np.bincount(y) / len(y)
        return 1.0 - np.sum(probs**2)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        return self._traverse(
            x, node.left if x[node.feature] <= node.threshold else node.right
        )


class CustomRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees, self.max_depth, self.min_samples_split = (
            n_trees,
            max_depth,
            min_samples_split,
        )
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        tree_preds = np.swapaxes([t.predict(X) for t in self.trees], 0, 1)
        return np.array([Counter(p).most_common(1)[0][0] for p in tree_preds])


# ==========================================
# 1. Models (Assume classes from previous logic are defined here)
# ==========================================


def run_single_outer_fold(fold_idx, train_idx, test_idx, X, y, configs, inner_cv):
    """
    This function runs on a single CPU core.
    It handles the scaling, tuning, and evaluation for ONE outer fold.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scaling inside the fold to prevent leakage
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma

    fold_results = {}

    for name, config in configs.items():
        # IMPORTANT: n_jobs=1 here because the OUTER loop handles parallelism
        grid = GridSearchCV(
            config["model"], config["params"], cv=inner_cv, scoring="accuracy", n_jobs=1
        )
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        fold_results[name] = {"score": score, "params": grid.best_params_}

    print(f"Finished Outer Fold {fold_idx + 1}")
    return fold_results


# ==========================================
# 2. Main Execution (The CPU Saturator)
# ==========================================

if __name__ == "__main__":
    # Load data
    X = np.load("data/X_train_final.npz")["data"]
    y = np.load("data/y_train.npz")["data"].astype(int)

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    configs = {
        "SVM": {"model": SVC(), "params": {"C": [1, 10], "kernel": ["linear", "rbf"]}},
        "LogReg": {
            "model": MultinomialLogisticRegression(),
            "params": {"reg_type": ["none", "l1", "l2"], "lambda_param": [0.1, 1, 5]},
        },
        "RandomForest": {
            "model": CustomRandomForest(),
            "params": {"n_trees": [20, 50, 70], "max_depth": [5, 10]},
        },
    }

    print(f"Starting Parallel Nested CV. Saturating CPU cores...")

    # The magic happens here: n_jobs=-1 parallelizes the OUTER folds.
    # If you have 5 folds, 5 cores will work at 100% immediately.
    raw_results = Parallel(n_jobs=-1)(
        delayed(run_single_outer_fold)(i, train, test, X, y, configs, inner_cv)
        for i, (train, test) in enumerate(outer_cv.split(X, y))
    )

    # 3. Aggregate Results
    final_report = {name: {"scores": [], "params": []} for name in configs}
    for fold_res in raw_results:
        for name in configs:
            final_report[name]["scores"].append(fold_res[name]["score"])
            final_report[name]["params"].append(fold_res[name]["params"])
