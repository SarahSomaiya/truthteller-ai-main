import pandas as pd
import numpy as np
import joblib
import os
import json
from collections import Counter

def entropy(labels):
    """Calculate entropy of labels"""
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    probs = [count / len(labels) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def information_gain(X, y, feature_index, threshold):
    """Calculate information gain for a split"""
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    left_y = y[left_mask]
    right_y = y[right_mask]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    total_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)

    left_weight = len(left_y) / len(y)
    right_weight = len(right_y) / len(y)

    return total_entropy - (left_weight * left_entropy + right_weight * right_entropy)

def find_best_split(X, y):
    """Find the best feature and threshold for splitting"""
    best_gain = 0
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for feature in range(n_features):
        # Try different thresholds
        unique_values = np.unique(X[:, feature])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for threshold in thresholds:
            gain = information_gain(X, y, feature, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        # Stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            # Return majority class
            return Counter(y).most_common(1)[0][0]

        # Find best split
        best_feature, best_threshold, best_gain = find_best_split(X, y)

        if best_gain == 0:
            # No good split found
            return Counter(y).most_common(1)[0][0]

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.fit(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def train(self, X, y):
        self.tree = self.fit(X, y)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample"""
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _random_feature_subset(self, X):
        """Select random subset of features"""
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        return feature_indices

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)

            # Random feature subset
            feature_indices = self._random_feature_subset(X_boot)
            X_boot_subset = X_boot[:, feature_indices]

            # Train tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.train(X_boot_subset, y_boot)

            # Store tree and its feature indices
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # Get predictions from all trees
        tree_predictions = []

        for tree, feature_indices in self.trees:
            X_subset = X[:, feature_indices]
            tree_predictions.append(tree.predict(X_subset))

        tree_predictions = np.array(tree_predictions)

        # Majority vote
        final_predictions = []
        for i in range(X.shape[0]):
            sample_predictions = tree_predictions[:, i]
            prediction = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(prediction)

        return np.array(final_predictions)

def train_advanced_model():
    """
    Train an advanced model using the engineered features
    """
    print("Loading dataset...")
    df = pd.read_csv('AuthentiText_X_2026_AI_vs_Human_Detection_1K.csv')

    # Use the engineered features
    feature_columns = [
        'perplexity_score', 'burstiness_index', 'syntactic_variability',
        'semantic_coherence_score', 'lexical_diversity_ratio',
        'readability_grade_level', 'generation_confidence_score'
    ]

    X = df[feature_columns].values
    y = (df['author_type'] == 'AI').astype(int).values  # 1 for AI, 0 for Human

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)} (AI: {np.sum(y)}, Human: {len(y) - np.sum(y)})")

    # Split the data (80/20)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Normalize features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    print("\nTraining Random Forest model...")

    # Train Random Forest
    rf = RandomForest(n_trees=50, max_depth=15, min_samples_split=5)
    rf.fit(X_train_scaled, y_train)

    # Make predictions
    train_predictions = rf.predict(X_train_scaled)
    test_predictions = rf.predict(X_test_scaled)

    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Calculate additional metrics
    tp = np.sum((test_predictions == 1) & (y_test == 1))
    fp = np.sum((test_predictions == 1) & (y_test == 0))
    fn = np.sum((test_predictions == 0) & (y_test == 1))
    tn = np.sum((test_predictions == 0) & (y_test == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\nTest Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"True Positives (AI correctly identified): {tp}")
    print(f"True Negatives (Human correctly identified): {tn}")
    print(f"False Positives (Human misclassified as AI): {fp}")
    print(f"False Negatives (AI misclassified as Human): {fn}")

    # Save model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)

    model_data = {
        'model': rf,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'feature_columns': feature_columns,
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Save using joblib
    joblib.dump(model_data, 'models/advanced_model.pkl')

    print("Advanced model saved successfully!")
    return model_data

if __name__ == "__main__":
    train_advanced_model()