import sys
from pathlib import Path
import joblib
import numpy as np
from gait_utils import FastGaitAnalyzer, extract_features_from_stats

if __name__ == "__main__":
    VIDEO_PATH = "./dataset/test/hemiplegic/gaitfront.mp4"

    MODEL_PATH = "./model/pose_landmarker.task"
    SAVE_DIR = Path("./saved_models")

    model_path = SAVE_DIR / "gait_svm_model.pkl"
    scaler_path = SAVE_DIR / "gait_scaler.pkl"

    if not (model_path.exists() and scaler_path.exists()):
        print("Error: Saved model files not found. Please run train.py first.")
        sys.exit(1)

    video_file = Path(VIDEO_PATH)
    if not video_file.exists():
        print(f"Error: Video file not found at {VIDEO_PATH}")
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"Analyzing video: {video_file.name} ...")
    with FastGaitAnalyzer(MODEL_PATH, resize_scale=50, min_visibility=0.6) as analyzer:
        stats = analyzer.analyze_video(video_file, verbose=True, show_video=True)
        features = extract_features_from_stats(stats)

    print("\n" + "=" * 50)
    if features is None:
        print("Prediction Failed: Insufficient valid poses detected in the video.")
    else:
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)

        prediction = model.predict(features_scaled)[0]
        decision_score = model.decision_function(features_scaled)[0]

        labels_map = {0: "Healthy", 1: "Hemiplegic"}
        result = labels_map[prediction]

        confidence_direction = (
            "Hemiplegic side" if decision_score > 0 else "Healthy side"
        )

        print("DETECTION RESULT:")
        print(f"  • Classification : {result}")
        print(
            f"  • Confidence Score: {abs(decision_score):.4f} (leaning towards the {confidence_direction})"
        )
    print("=" * 50)
