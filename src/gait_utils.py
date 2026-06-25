from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def calculate_joint_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


class GaitAnalyzer:
    def __init__(self, min_visibility=0.5):
        self.min_visibility = min_visibility
        self.left_knee_history = []
        self.right_knee_history = []
        self.left_foot_history = []
        self.right_foot_history = []

    def analyze_gait_cycle(self, world_landmarks, timestamp_ms, verbose=True):
        if world_landmarks is None:
            return False
        indices = [23, 25, 27, 31, 24, 26, 28, 32]
        if any(world_landmarks[i].visibility <= self.min_visibility for i in indices):
            return False

        left_hip, left_knee, left_ankle, left_foot = [
            world_landmarks[i] for i in indices[:4]
        ]
        right_hip, right_knee, right_ankle, right_foot = [
            world_landmarks[i] for i in indices[4:]
        ]

        self.left_knee_history.append(
            calculate_joint_angle(left_hip, left_knee, left_ankle)
        )
        self.right_knee_history.append(
            calculate_joint_angle(right_hip, right_knee, right_ankle)
        )
        self.left_foot_history.append(
            calculate_joint_angle(left_knee, left_ankle, left_foot)
        )
        self.right_foot_history.append(
            calculate_joint_angle(right_knee, right_ankle, right_foot)
        )

        if verbose:
            print(f"[{timestamp_ms}ms] Processing frame angles...", end="\r")
        return True

    def get_summary_statistics(self):
        if not self.left_knee_history or not self.left_foot_history:
            return None
        lk, rk = np.array(self.left_knee_history), np.array(self.right_knee_history)
        lf, rf = np.array(self.left_foot_history), np.array(self.right_foot_history)
        return {
            "left": {
                "knee_min": float(lk.min()),
                "knee_max": float(lk.max()),
                "knee_avg": float(lk.mean()),
                "knee_std": float(lk.std()),
                "foot_min": float(lf.min()),
                "foot_max": float(lf.max()),
                "foot_avg": float(lf.mean()),
                "foot_std": float(lf.std()),
            },
            "right": {
                "knee_min": float(rk.min()),
                "knee_max": float(rk.max()),
                "knee_avg": float(rk.mean()),
                "knee_std": float(rk.std()),
                "foot_min": float(rf.min()),
                "foot_max": float(rf.max()),
                "foot_avg": float(rf.mean()),
                "foot_std": float(rf.std()),
            },
        }


class FastGaitAnalyzer:
    def __init__(self, model_path, resize_scale=50, min_visibility=0.6):
        self.model_path = str(model_path)
        self.resize_scale = resize_scale
        self.min_visibility = min_visibility
        self.global_timestamp_offset_ms = 0
        self.detector = self.init_detector()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        analyzer = GaitAnalyzer(min_visibility=self.min_visibility)
        frame_counter = 0
        last_frame_timestamp_ms = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_counter += 1
            frame = self.preprocess_frame(frame)
            local_time_ms = int((frame_counter - 1) * 1000 / fps) if fps > 0 else 0
            absolute_timestamp_ms = self.global_timestamp_offset_ms + local_time_ms
            last_frame_timestamp_ms = local_time_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = self.detector.detect_for_video(
                mp_image, absolute_timestamp_ms
            )

            if detection_result.pose_world_landmarks:
                analyzer.analyze_gait_cycle(
                    detection_result.pose_world_landmarks[0],
                    absolute_timestamp_ms,
                    verbose=verbose,
                )

        cap.release()
        self.global_timestamp_offset_ms += last_frame_timestamp_ms + 1
        return analyzer.get_summary_statistics()

    def close(self):
        if self.detector:
            self.detector.close()


def extract_features_from_stats(stats):
    if stats is None:
        return None
    left_knee_rom = stats["left"]["knee_max"] - stats["left"]["knee_min"]
    right_knee_rom = stats["right"]["knee_max"] - stats["right"]["knee_min"]
    eps = 1e-6
    knee_rom_si = (
        abs(left_knee_rom - right_knee_rom)
        / ((left_knee_rom + right_knee_rom) / 2 + eps)
    ) * 100
    foot_min_si = (
        abs(stats["left"]["foot_min"] - stats["right"]["foot_min"])
        / ((stats["left"]["foot_min"] + stats["right"]["foot_min"]) / 2 + eps)
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


def load_dataset_from_folders(
    dataset_root, analyzer, file_extensions=(".webm", ".mp4", ".avi")
):
    X, y, filenames = [], [], []
    label_map = {"healthy": 0, "hemiplegic": 1}
    dataset_path = Path(dataset_root)

    for label_name, label_id in label_map.items():
        folder = dataset_path / label_name
        if not folder.is_dir():
            continue
        video_files = [
            p for p in folder.iterdir() if p.suffix.lower() in file_extensions
        ]

        for video_path in sorted(video_files):
            print(f"Processing {label_name}: {video_path.name}")
            stats = analyzer.analyze_video(video_path, verbose=False)
            features = extract_features_from_stats(stats)
            if features is None:
                continue
            X.append(features)
            y.append(label_id)
            filenames.append(str(video_path))
    return np.array(X), np.array(y), filenames
