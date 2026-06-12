import glob
import os
import time
import warnings

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -------------------------------------------------------------------
# 1. Core geometry & analysis classes
# -------------------------------------------------------------------


def calculate_joint_angle(p1, p2, p3):
    """Calculates the 3D angle at the joint vertex p2 using vectors."""
    hip = np.array([p1.x, p1.y, p1.z])
    knee = np.array([p2.x, p2.y, p2.z])
    ankle = np.array([p3.x, p3.y, p3.z])

    v1 = hip - knee
    v2 = ankle - knee

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


class GaitAnalyzer:
    """Collects per‑frame joint angles and computes summary statistics."""

    def __init__(self, min_visibility=0.5):
        self.min_visibility = min_visibility
        self.left_knee_history = []
        self.right_knee_history = []
        self.left_foot_history = []
        self.right_foot_history = []

    def analyze_gait_cycle(self, world_landmarks, timestamp_ms):
        if world_landmarks is None:
            return False

        left_hip = world_landmarks[23]
        left_knee = world_landmarks[25]
        left_ankle = world_landmarks[27]
        left_foot = world_landmarks[31]

        right_hip = world_landmarks[24]
        right_knee = world_landmarks[26]
        right_ankle = world_landmarks[28]
        right_foot = world_landmarks[32]

        all_visible = (
            left_hip.visibility > self.min_visibility
            and left_knee.visibility > self.min_visibility
            and left_ankle.visibility > self.min_visibility
            and left_foot.visibility > self.min_visibility
            and right_hip.visibility > self.min_visibility
            and right_knee.visibility > self.min_visibility
            and right_ankle.visibility > self.min_visibility
            and right_foot.visibility > self.min_visibility
        )

        if all_visible:
            left_knee_angle = calculate_joint_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_joint_angle(right_hip, right_knee, right_ankle)
            left_foot_angle = calculate_joint_angle(left_knee, left_ankle, left_foot)
            right_foot_angle = calculate_joint_angle(
                right_knee, right_ankle, right_foot
            )

            self.left_knee_history.append(left_knee_angle)
            self.right_knee_history.append(right_knee_angle)
            self.left_foot_history.append(left_foot_angle)
            self.right_foot_history.append(right_foot_angle)

            print(
                f"[{timestamp_ms}ms] KNEES L: {left_knee_angle:.1f}° R: {right_knee_angle:.1f}° "
                f"| FEET L: {left_foot_angle:.1f}° R: {right_foot_angle:.1f}°",
                end="\r",
            )
            return True
        else:
            return False

    def get_summary_statistics(self):
        if not self.left_knee_history or not self.left_foot_history:
            return None

        return {
            "left": {
                "knee_min": float(np.min(self.left_knee_history)),
                "knee_max": float(np.max(self.left_knee_history)),
                "knee_avg": float(np.mean(self.left_knee_history)),
                "knee_std": float(np.std(self.left_knee_history)),
                "foot_min": float(np.min(self.left_foot_history)),
                "foot_max": float(np.max(self.left_foot_history)),
                "foot_avg": float(np.mean(self.left_foot_history)),
                "foot_std": float(np.std(self.left_foot_history)),
            },
            "right": {
                "knee_min": float(np.min(self.right_knee_history)),
                "knee_max": float(np.max(self.right_knee_history)),
                "knee_avg": float(np.mean(self.right_knee_history)),
                "knee_std": float(np.std(self.right_knee_history)),
                "foot_min": float(np.min(self.right_foot_history)),
                "foot_max": float(np.max(self.right_foot_history)),
                "foot_avg": float(np.mean(self.right_foot_history)),
                "foot_std": float(np.std(self.right_foot_history)),
            },
        }


class FastGaitAnalyzer:
    """Efficient, single‑threaded video pose estimator using MediaPipe."""

    def __init__(self, model_path, skip_frames=2, resize_scale=50, min_visibility=0.6):
        self.model_path = model_path
        self.skip_frames = skip_frames
        self.resize_scale = resize_scale
        self.min_visibility = min_visibility

        # FIX: Track absolute time offset across distinct file segments
        self.global_timestamp_offset_ms = 0

        # Detector is initialized once here to protect system lifecycle boundaries
        self.detector = self.init_detector()

    def init_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def preprocess_frame(self, frame):
        if self.resize_scale != 100:
            width = int(frame.shape[1] * self.resize_scale / 100)
            height = int(frame.shape[0] * self.resize_scale / 100)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        return frame

    def analyze_video(self, video_path, verbose=True):
        """Process a video and return summary statistics."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        analyzer = GaitAnalyzer(min_visibility=self.min_visibility)

        frame_counter = 0
        processed_count = 0
        last_frame_timestamp_ms = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1

            if frame_counter % (self.skip_frames + 1) != 0:
                continue

            frame = self.preprocess_frame(frame)

            # Calculate local video timestamp
            local_time_ms = int((frame_counter - 1) * 1000 / fps) if fps > 0 else 0

            # FIX: Compound onto global pipeline runtime to force strictly monotonic increases
            absolute_timestamp_ms = self.global_timestamp_offset_ms + local_time_ms
            last_frame_timestamp_ms = local_time_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = self.detector.detect_for_video(
                mp_image, absolute_timestamp_ms
            )

            if detection_result.pose_world_landmarks:
                world_landmarks = detection_result.pose_world_landmarks[0]
                analyzer.analyze_gait_cycle(world_landmarks, absolute_timestamp_ms)
                processed_count += 1

        cap.release()

        # FIX: Increment offset boundary by last frame runtime + 1ms padding gap for the next video asset
        self.global_timestamp_offset_ms += last_frame_timestamp_ms + 1

        return analyzer.get_summary_statistics()

    def close(self):
        """Explicit cleanup method invoked outside loop contexts."""
        if self.detector:
            self.detector.close()


# -------------------------------------------------------------------
# 2. Feature extraction (pure function)
# -------------------------------------------------------------------


def extract_features_from_stats(stats):
    """Convert a stats dictionary into an 8‑dimensional feature vector."""
    if stats is None:
        return None

    left_knee_rom = stats["left"]["knee_max"] - stats["left"]["knee_min"]
    right_knee_rom = stats["right"]["knee_max"] - stats["right"]["knee_min"]

    # Normalized Symmetry Index Metrics
    knee_rom_si = (
        abs(left_knee_rom - right_knee_rom) / ((left_knee_rom + right_knee_rom) / 2)
    ) * 100

    foot_min_si = (
        abs(stats["left"]["foot_min"] - stats["right"]["foot_min"])
        / ((stats["left"]["foot_min"] + stats["right"]["foot_min"]) / 2)
    ) * 100

    return [
        left_knee_rom,
        right_knee_rom,
        stats["left"]["knee_std"],
        stats["right"]["knee_std"],
        stats["left"]["foot_min"],
        stats["right"]["foot_min"],
        knee_rom_si,
        foot_min_si,
    ]


# -------------------------------------------------------------------
# 3. Dataset loading & label assignment
# -------------------------------------------------------------------


def load_dataset_from_folders(
    dataset_root, analyzer, file_extensions=("*.webm", "*.mp4")
):
    """Walk through dataset directories, process videos, and return features."""
    X = []
    y = []
    filenames = []

    label_map = {
        "healthy": 0,
        "hemiplegic": 1,
    }

    for label_name, label_id in label_map.items():
        folder = os.path.join(dataset_root, label_name)
        if not os.path.isdir(folder):
            warnings.warn(f"Folder not found: {folder}")
            continue

        video_files = []
        for ext in file_extensions:
            video_files.extend(glob.glob(os.path.join(folder, ext)))

        for video_path in sorted(video_files):
            print(f"Processing {label_name} video: {os.path.basename(video_path)}")
            stats = analyzer.analyze_video(video_path, verbose=False)
            features = extract_features_from_stats(stats)

            if features is None:
                warnings.warn(f"Skipping {video_path} – insufficient valid poses")
                continue

            X.append(features)
            y.append(label_id)
            filenames.append(video_path)

    return np.array(X), np.array(y), filenames


# -------------------------------------------------------------------
# 4. SVM training pipeline
# -------------------------------------------------------------------


def train_gait_svm(X, y, test_size=0.3, cv_folds=5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    _, counts = np.unique(y_train, return_counts=True)
    min_samples_per_class = np.min(counts)
    effective_cv = min(cv_folds, min_samples_per_class)
    if effective_cv < 2:
        raise ValueError("Need at least 2 samples per class for cross‑validation.")
    if effective_cv < cv_folds:
        print(f"⚠ Reduced CV folds to {effective_cv} due to small class size.")

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
    }

    base_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=random_state,
    )
    grid = GridSearchCV(base_svm, param_grid, cv=effective_cv, scoring="accuracy")
    grid.fit(X_train_scaled, y_train)

    print("\nBest parameters:", grid.best_params_)
    print(f"Best cross‑validation accuracy: {grid.best_score_ * 100:.1f}%")

    y_pred = grid.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc * 100:.1f}%")
    print("\nClassification report (test set):")
    print(
        classification_report(
            y_test, y_pred, target_names=["healthy", "hemiplegic"], zero_division="0.0"
        )
    )

    return grid.best_estimator_, scaler, (X_test_scaled, y_test, y_pred)


# -------------------------------------------------------------------
# 5. Main entry point (Dedicated Training Loop)
# -------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "./model/pose_landmarker.task"
    DATASET_ROOT = "./dataset"
    SKIP_FRAMES = 2
    RESIZE_SCALE = 50
    MIN_VISIBILITY = 0.6

    # Graph environment initializes once safely
    analyzer = FastGaitAnalyzer(
        model_path=MODEL_PATH,
        skip_frames=SKIP_FRAMES,
        resize_scale=RESIZE_SCALE,
        min_visibility=MIN_VISIBILITY,
    )

    try:
        print("=" * 60)
        print("TRAINING PIPELINE: Loading dataset and training SVM ...")
        print("=" * 60)

        X, y, video_files = load_dataset_from_folders(DATASET_ROOT, analyzer)

        if len(X) < 4:
            print(
                f"\nDataset contains only {len(X)} processed elements. Add more files to compute features."
            )
        else:
            print(f"\nDataset successfully built: {len(X)} samples loaded.")
            print(
                "Distribution mappings:",
                dict(zip(*np.unique(y, return_counts=True))),
            )

            best_model, scaler, test_data = train_gait_svm(X, y)

            # --- PERSISTENCE EXPORT LAYER ---
            os.makedirs("./saved_models", exist_ok=True)
            joblib.dump(best_model, "./saved_models/gait_svm_model.pkl")
            joblib.dump(scaler, "./saved_models/gait_scaler.pkl")

            print("\n" + "═" * 50)
            print("💾 Pipeline modules safely exported to file structures:")
            print("  • Model Architecture : ./saved_models/gait_svm_model.pkl")
            print("  • Scaling Parameters  : ./saved_models/gait_scaler.pkl")
            print("═" * 50)

    finally:
        print("\nClosing MediaPipe task managers...")
        analyzer.close()
        print("Done.")
