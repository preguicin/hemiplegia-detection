from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from gait_utils import FastGaitAnalyzer, load_dataset_from_folders


def train_gait_svm(X_train, y_train, cv_folds=5, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    _, counts = np.unique(y_train, return_counts=True)
    effective_cv = max(2, min(cv_folds, np.min(counts)))

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
    }

    base_svm = SVC(kernel="rbf", class_weight="balanced", random_state=random_state)
    grid = GridSearchCV(base_svm, param_grid, cv=effective_cv, scoring="accuracy")
    grid.fit(X_train_scaled, y_train)

    print("\nBest CV Parameters:", grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_ * 100:.1f}%")
    return grid.best_estimator_, scaler


if __name__ == "__main__":
    MODEL_PATH = "./model/pose_landmarker.task"
    TRAIN_DATASET_ROOT = "./dataset/train"

    with FastGaitAnalyzer(MODEL_PATH, resize_scale=50, min_visibility=0.6) as analyzer:
        print("--- Extracting Training Features ---")
        X_train, y_train, _ = load_dataset_from_folders(TRAIN_DATASET_ROOT, analyzer)

    if len(X_train) < 4:
        print("Insufficient training samples.")
    else:
        best_model, scaler = train_gait_svm(X_train, y_train)

        save_dir = Path("./saved_models")
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, save_dir / "gait_svm_model.pkl")
        joblib.dump(scaler, save_dir / "gait_scaler.pkl")
        print(f"\nModel and scaler successfully saved to {save_dir}")
