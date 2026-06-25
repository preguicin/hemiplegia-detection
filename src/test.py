from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report
from gait_utils import FastGaitAnalyzer, load_dataset_from_folders

if __name__ == "__main__":
    MODEL_PATH = "./model/pose_landmarker.task"
    TEST_DATASET_ROOT = "./dataset/test"
    SAVE_DIR = Path("./saved_models")

    model_path = SAVE_DIR / "gait_svm_model.pkl"
    scaler_path = SAVE_DIR / "gait_scaler.pkl"

    if not (model_path.exists() and scaler_path.exists()):
        raise FileNotFoundError("Saved model files not found. Run train.py first.")

    # 1. Load the pre-trained assets
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Pre-trained model and scaler assets loaded successfully.")

    # 2. Extract features from the independent test set folder
    with FastGaitAnalyzer(MODEL_PATH, resize_scale=50, min_visibility=0.6) as analyzer:
        print("\n--- Extracting Testing Features ---")
        X_test, y_test, _ = load_dataset_from_folders(TEST_DATASET_ROOT, analyzer)

    if len(X_test) == 0:
        print("No test samples were processed successfully.")
    else:
        # 3. Scale test data using the training parameters, then evaluate
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        print(f"\nFinal Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
        print("\nClassification Report (Test Data):")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["healthy", "hemiplegic"],
                zero_division=0.0,
            )
        )
