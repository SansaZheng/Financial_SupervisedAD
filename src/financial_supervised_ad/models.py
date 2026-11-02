"""Classical algorithms through maintained reference implementations.

Hyperparameters are fixed a priori, not tuned against test labels.
"""
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

METHODS = ("logistic_regression", "gaussian_nb", "knn", "svm_rbf", "decision_tree",
           "random_forest", "adaboost", "gradient_boosting", "xgboost", "mlp")


def build_model(name, seed=42, n_jobs=1, quick=False):
    trees = 30 if quick else 100
    factories = {
        "logistic_regression": lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        "gaussian_nb": lambda: GaussianNB(var_smoothing=1e-9),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=n_jobs),
        # A margin score is sufficient for ranking and validation thresholding;
        # disabling internal probability CV avoids calibration cost/leakage questions.
        "svm_rbf": lambda: SVC(C=1.0, kernel="rbf", gamma="scale", probability=False,
                                cache_size=256, random_state=seed),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=trees, min_samples_leaf=2,
                                                        random_state=seed, n_jobs=n_jobs),
        "adaboost": lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
            n_estimators=trees, learning_rate=0.5, random_state=seed),
        "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=trees,
            learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=100 if quick else 300, early_stopping=False, random_state=seed),
    }
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=trees, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            random_state=seed, n_jobs=n_jobs)
    if name not in factories:
        raise ValueError(f"Unknown method {name!r}; choose from {METHODS}")
    return factories[name]()
