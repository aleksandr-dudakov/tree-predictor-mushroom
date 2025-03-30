import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split, KFold
from graphviz import Digraph
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(89)

# Fetch dataset from UCI repository
secondary_mushroom = fetch_ucirepo(id=848)

# Load features and target as DataFrames
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

# Fill missing values in categorical columns with "unknown"
#X = X.apply(lambda col: col.fillna("unknown") if col.dtype == object else col)

# Convert all object columns to categorical for efficiency
#for col in X.select_dtypes(include=['object']).columns:
#    X[col] = X[col].astype('category')

X = pd.get_dummies(X).astype('int')

# Convert target to categorical
y = y.astype('category')
y = y.squeeze()

# Split data into 80% training and 20% testing (with stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=89)

class Node:
    def __init__(self, decision_fn=None, value=None, left=None, right=None, test_feature=None, test_value=None):
        self.decision_fn = decision_fn    # Decision function returning a Boolean
        self.value = value                # Prediction if this is a leaf
        self.left = left                  # Left child node
        self.right = right                # Right child node
        self.test_feature = test_feature  # The feature used for the test at this node
        self.test_value = test_value      # The threshold (for numerical) or category (for categorical) used
        self.is_leaf = value is not None  # True if node is a leaf

class TreePredictor:
    # Initialize with the chosen psi-function and stopping criteria.
    def __init__(self, psi="gini", max_depth=None, min_samples_split=None, min_impurity_decrease=None, max_features=None):
        self.psi = psi
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.root = None

    # Compute training error contribution using the chosen psi-function
    # Assumes binary classification
    def _impurity(self, y):
        # p is the frequency of the first label (psi is symmetric so the choice does not matter)
        p = y.value_counts().iloc[0] / len(y)
        if self.psi == "min":
            return min(p, 1 - p)
        elif self.psi == "gini":
            return 2 * p * (1 - p)
        elif self.psi == "scaled_entropy":
            return -0.5 * (p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9))
        elif self.psi == "sqrt":
            return np.sqrt(p * (1 - p))
        else:
            raise ValueError("Unknown psi function. Choose 'min', 'gini', 'scaled_entropy', or 'sqrt'.")

    # Find the best split based on the decrease in training error
    def _best_split(self, X, y):
        best_decrease = 0.0
        best_feature = None
        best_decision_fn = None
        best_left_idx = None
        best_right_idx = None
        best_test_value = None
        current_error = self._impurity(y)
        n_samples = len(y)
        
        # Try features in X
        # Use random feature sampling if max_features is set
        features = list(X.columns)
        if self.max_features is not None and self.max_features < len(features):
            features = np.random.choice(features, size=self.max_features, replace=False)
        for feature in features:
            X_feature = X[feature]
            # For numerical features
            if pd.api.types.is_numeric_dtype(X_feature):
                thresholds = np.sort(X_feature.unique())
                for threshold in thresholds:
                    left_idx = X_feature < threshold
                    right_idx = ~left_idx
                    # Skip splits that yield empty branches
                    if left_idx.sum() == 0 or right_idx.sum() == 0:
                        continue
                    error_left = self._impurity(y[left_idx])
                    error_right = self._impurity(y[right_idx])
                    weighted_error = (left_idx.sum() * error_left + right_idx.sum() * error_right) / n_samples
                    decrease = current_error - weighted_error
                    if decrease > best_decrease:
                        best_decrease = decrease
                        best_feature = feature
                        best_decision_fn = lambda x, thresh=threshold, feat=feature: x[feat] < thresh
                        best_left_idx = left_idx
                        best_right_idx = right_idx
                        best_test_value = threshold
            else:
                # For categorical features, assume the column is already categorical
                cats = X_feature.cat.categories
                for category in cats:
                    left_idx = X_feature == category
                    right_idx = ~left_idx
                    if left_idx.sum() == 0 or right_idx.sum() == 0:
                        continue
                    error_left = self._impurity(y[left_idx])
                    error_right = self._impurity(y[right_idx])
                    weighted_error = (left_idx.sum() * error_left + right_idx.sum() * error_right) / n_samples
                    decrease = current_error - weighted_error
                    if decrease > best_decrease:
                        best_decrease = decrease
                        best_feature = feature
                        best_decision_fn = lambda x, cat=category, feat=feature: x[feat] == cat
                        best_left_idx = left_idx
                        best_right_idx = right_idx
                        best_test_value = category
        return best_feature, best_decision_fn, best_left_idx, best_right_idx, best_decrease, best_test_value

    # Recursively build the tree predictor
    def _build_tree(self, X, y, depth=0):
        # Apply stopping criteria: max_depth, minimum samples, or pure leaf (zero training error)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (self.min_samples_split is not None and len(y) < self.min_samples_split) or \
           (self._impurity(y) == 0):
            # Label the leaf with the majority class (ties broken arbitrarily)
            majority_class = y.mode().iat[0]
            return Node(value=majority_class)
        
        best_feature, decision_fn, left_idx, right_idx, error_decrease, best_test_value = self._best_split(X, y)
        # Stop if no valid split is found or if the error decrease is too small.
        if best_feature is None or (self.min_impurity_decrease is not None and error_decrease < self.min_impurity_decrease):
            majority_class = y.mode().iat[0]
            return Node(value=majority_class)
        
        # Recursively build subtrees for left and right splits.
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(decision_fn=decision_fn, left=left_tree, right=right_tree, 
                    test_feature=best_feature, test_value=best_test_value)

    # Train the tree predictor on a training set
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    # Helper to predict the label for a single data point
    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if node.decision_fn(x):
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    # Predict labels for all examples in dataset X
    def predict(self, X):
        # Convert DataFrame rows to a list of dictionaries for faster iteration
        records = X.to_dict(orient='records')
        # Use a list comprehension to apply the recursive prediction to each record
        predictions = [self._predict_one(record, self.root) for record in records]
        return pd.Series(predictions, index=X.index)

    # Evaluate the predictor on a dataset using 0–1 loss
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)

    # Graphical visualization of the tree using Graphviz
    def visualize(self, filename=None, view=True):
        dot = Digraph()
        
        # Helper function to add nodes recursively
        def _add_nodes(node, counter):
            # Get a unique id for the current node
            node_id = str(next(counter))
            # Build label for the node.
            if node.is_leaf:
                label = f"Leaf: {node.value}"
            else:
                # Use "<" for numeric tests, "==" for categorical tests
                op = "<" if isinstance(node.test_value, (int, float, np.integer, np.floating)) else "=="
                label = f"x[{node.test_feature}] {op} {node.test_value}"
            dot.node(node_id, label)
            # If not a leaf, add child nodes and edges
            if not node.is_leaf:
                left_id = _add_nodes(node.left, counter)
                dot.edge(node_id, left_id, label="True")
                right_id = _add_nodes(node.right, counter)
                dot.edge(node_id, right_id, label="False")
            return node_id
        
        # Create a counter for unique node ids
        counter = iter(range(10000))
        _add_nodes(self.root, counter)
        
        # If a filename is provided, render to file
        if filename:
            dot.render(filename, view=view)
        return dot

class RandomForestPredictor:
    def __init__(self, n_estimators=10, max_features=None, psi="gini", 
                 max_depth=None, min_samples_split=None, min_impurity_decrease=None,
                 use_oob=False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.psi = psi
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.use_oob = use_oob
        self.trees = []
        self.oob_indices = []  # To store OOB indices for each tree

    def fit(self, X, y):
        n_samples = len(y)
        self.trees = []
        self.oob_indices = []
        for i in range(self.n_estimators):
            # Create a bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            # OOB indices are those not selected in the bootstrap sample
            oob_idx = np.setdiff1d(np.arange(n_samples), indices)
            self.oob_indices.append(oob_idx)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            tree = TreePredictor(psi=self.psi, max_depth=self.max_depth, 
                                 min_samples_split=self.min_samples_split,
                                 min_impurity_decrease=self.min_impurity_decrease, 
                                 max_features=self.max_features)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from all trees
        predictions = pd.DataFrame({i: tree.predict(X) for i, tree in enumerate(self.trees)})
        return predictions.mode(axis=1)[0]

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds != y)

    def compute_oob_error(self, X, y):
        """Compute the OOB error estimate using the trees for which a sample was out-of-bag."""
        n_samples = len(y)
        # Dictionary to accumulate votes for each sample
        votes = {i: [] for i in range(n_samples)}
        for tree, oob_idx in zip(self.trees, self.oob_indices):
            if len(oob_idx) == 0:
                continue
            # Get predictions only for the OOB samples of this tree
            preds = tree.predict(X.iloc[oob_idx])
            for idx, pred in zip(oob_idx, preds):
                votes[idx].append(pred)
        # Compute majority vote for each sample that has any OOB prediction
        oob_predictions = {}
        for i, vote_list in votes.items():
            if vote_list:
                # Majority vote (ties are broken arbitrarily)
                oob_predictions[i] = max(set(vote_list), key=vote_list.count)
        # Evaluate OOB error only on samples with at least one vote
        valid_idx = list(oob_predictions.keys())
        if len(valid_idx) == 0:
            return None
        error = np.mean([oob_predictions[i] != y.iloc[i] for i in valid_idx])
        return error

def baseline_experiments(X, y, psi_list, grid, parameter):
    # This function runs baseline experiments on the full training set
    rows = []
    for psi in psi_list:
        for val in grid:
            # Set the corresponding parameter in the model
            if parameter == "max_depth":
                model = TreePredictor(psi=psi, max_depth=val)
            elif parameter == "min_impurity_decrease":
                model = TreePredictor(psi=psi, min_impurity_decrease=val)
            elif parameter == "min_samples_split":
                model = TreePredictor(psi=psi, min_samples_split=val)
            else:
                raise ValueError("Unsupported parameter. Use 'max_depth', 'min_impurity_decrease' or 'min_samples_split'.")
            model.fit(X, y)
            loss = model.evaluate(X, y)
            rows.append({'psi': psi, parameter: val, 'loss': loss})
            print({'psi': psi, parameter: val, 'loss': loss})
    return pd.DataFrame(rows)

def cv_experiments(X, y, psi_list, grid, parameter, k=5, random_state=89):
    # This function performs k-fold cross-validation experiments for hyperparameter tuning
    rows = []
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    for psi in psi_list:
        for val in grid:
            cv_losses = []
            for train_idx, val_idx in kf.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                if parameter == "max_depth":
                    model = TreePredictor(psi=psi, max_depth=val)
                elif parameter == "min_impurity_decrease":
                    model = TreePredictor(psi=psi, min_impurity_decrease=val)
                elif parameter == "min_samples_split":
                    model = TreePredictor(psi=psi, min_samples_split=val)
                else:
                    raise ValueError("Unsupported parameter. Use 'max_depth', 'min_impurity_decrease' or 'min_samples_split'.")
                model.fit(X_train_cv, y_train_cv)
                cv_losses.append(model.evaluate(X_val_cv, y_val_cv))
            rows.append({'psi': psi, parameter: val, 'cv_loss': np.mean(cv_losses)})
            print({'psi': psi, parameter: val, 'cv_loss': np.mean(cv_losses)})
    return pd.DataFrame(rows)

def rf_oob_experiments(X, y, max_features_grid, n_estimators=50, psi="gini", max_depth=35, 
                       min_samples_split=2, min_impurity_decrease=None):
    rows = []
    for max_feat in max_features_grid:
        rf = RandomForestPredictor(n_estimators=n_estimators, max_features=max_feat, psi=psi, 
                                   max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_impurity_decrease=min_impurity_decrease, use_oob=True)
        rf.fit(X, y)
        oob_error = rf.compute_oob_error(X, y)
        rows.append({'max_features': max_feat, 'oob_error': oob_error})
        print({'max_features': max_feat, 'oob_error': oob_error})
    return pd.DataFrame(rows)
    
def main():
    ## Decision Tree
    # Parameters grid
    psi_list = ["gini", "scaled_entropy", "sqrt"]
    max_depth_grid = [15, 20, 25, 30, 35, 40, 45]
    min_impurity_grid = [0.0, 0.005, 0.01]
    
    # Baseline experiments
    df_baseline_max_depth = baseline_experiments(X_train, y_train, psi_list, max_depth_grid, "max_depth")
    df_baseline_min_impurity = baseline_experiments(X_train, y_train, psi_list, min_impurity_grid, "min_impurity_decrease")
    
    # Cross-validation experiments (hyperparameter tuning)
    df_cv_max_depth = cv_experiments(X_train, y_train, psi_list, max_depth_grid, "max_depth")
    df_cv_min_impurity = cv_experiments(X_train, y_train, psi_list, min_impurity_grid, "min_impurity_decrease")
    
    # Choose best model from CV experiments (max_depth grid)
    best_idx = df_cv_max_depth['cv_loss'].idxmin()
    best_params = df_cv_max_depth.iloc[best_idx]
    print("Best model parameters from CV (max_depth tuning):")
    print(best_params)
    
    # Retrain the model with these best parameters on the entire training set
    best_model = TreePredictor(psi=best_params['psi'], max_depth=best_params['max_depth'])
    best_model.fit(X_train, y_train)
    #best_model.visualize()
    
    # Evaluate the best model on the test set
    test_loss = best_model.evaluate(X_test, y_test)
    print("\nTest set loss:", test_loss)
    
    # Plot training loss vs. max_depth
    sns.lineplot(data=df_baseline_max_depth, x='max_depth', y='loss', hue='psi', marker='o')
    plt.title('Training Loss vs max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('Training Loss')
    plt.show()
    
    # Plot training loss vs. min_impurity_decrease
    sns.lineplot(data=df_baseline_min_impurity, x='min_impurity_decrease', y='loss', hue='psi', marker='o')
    plt.title('Training Loss vs min_impurity_decrease')
    plt.xlabel('min_impurity_decrease')
    plt.ylabel('Training Loss')
    plt.show()
    
    # Plot CV loss vs. max_depth
    sns.lineplot(data=df_cv_max_depth, x='max_depth', y='cv_loss', hue='psi', marker='o')
    plt.title('CV Loss vs max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('CV Loss')
    plt.show()
    
    # Plot CV loss vs. min_impurity_decrease
    sns.lineplot(data=df_cv_min_impurity, x='min_impurity_decrease', y='cv_loss', hue='psi', marker='o')
    plt.title('CV Loss vs min_impurity_decrease')
    plt.xlabel('min_impurity_decrease')
    plt.ylabel('CV Loss')
    plt.show()
    
    # Filter results for the best psi function only (as selected from CV)
    df_baseline_best = df_baseline_max_depth[df_baseline_max_depth['psi'] == best_params['psi']]
    df_cv_best = df_cv_max_depth[df_cv_max_depth['psi'] == best_params['psi']]
    
    # Plot of training loss for the best psi with the best model highlighted
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_baseline_best, x='max_depth', y='loss', marker='o', label='Training Loss')
    # Mark the best max_depth chosen from CV
    plt.axvline(x=best_params['max_depth'], color='red', linestyle='--', label=f'Best max_depth ({best_params["max_depth"]})')
    plt.title(f'Training Loss for Best Psi: {best_params["psi"]}')
    plt.xlabel('max_depth')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()
    
    # Combined plot: Training Loss vs. CV Loss (overfitting analysis) for the best psi
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_baseline_best, x='max_depth', y='loss', marker='o', label='Training Loss')
    sns.lineplot(data=df_cv_best, x='max_depth', y='cv_loss', marker='o', label='CV Loss')
    plt.axvline(x=best_params['max_depth'], color='red', linestyle='--', label=f'Best max_depth ({best_params["max_depth"]})')
    plt.title(f'Overfitting Analysis for Best Psi: {best_params["psi"]}')
    plt.xlabel('max_depth')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    ## Random Forest
    # Define grid for max_features (None means using all features)
    max_features_grid = [10, 15, 20, 30, 60, 90, 119]
    df_rf_oob = rf_oob_experiments(X_train, y_train, max_features_grid)
    
    # Find the best max_features value (min OOB error)
    best_idx = df_rf_oob['oob_error'].idxmin()
    best_row = df_rf_oob.iloc[best_idx]
    best_max_features = int(best_row['max_features'])
    min_oob_error = best_row['oob_error']
    print(f"Best max_features: {best_max_features}, with OOB error: {min_oob_error}")
    
    # Plot OOB loss vs. max_features with a line indicating the best max_features
    sns.lineplot(data=df_rf_oob, x='max_features', y='oob_error', marker='o')
    plt.axvline(x=best_max_features, color='red', linestyle='--', label=f'Best max_features = {best_max_features}')
    plt.title('Random Forest OOB Loss vs. max_features')
    plt.xlabel('max_features')
    plt.ylabel('OOB Error')
    plt.legend()
    plt.show()
    
    # Retrain the RandomForestPredictor using the best max_features on the full training set
    best_rf = RandomForestPredictor(max_features=best_max_features)
    best_rf.fit(X_train, y_train)
    
    # Evaluate the best model on the test set
    test_loss = best_rf.evaluate(X_test, y_test)
    print(f"\nTest set loss with best max_features ({best_max_features}): {test_loss}")

if __name__ == "__main__":
    main()