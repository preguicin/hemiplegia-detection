import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def calculate_joint_angle(p1, p2, p3):
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
    def __init__(self, min_visibility=0.5):
        self.min_visibility = min_visibility
        self.left_knee_history = []
        self.right_knee_history = []
        self.left_foot_history = []
        self.right_foot_history = []

    def analyze_gait_cycle(self, world_landmarks, timestamp_ms):
        if world_landmarks is None:
            return False

        # Lower Body Landmarks
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
                f"[{timestamp_ms}ms] KNEES L: {left_knee_angle:.1f}° R: {right_knee_angle:.1f}° | FEET L: {left_foot_angle:.1f}° R: {right_foot_angle:.1f}°",
                end="\r",
            )
            return True
        else:
            print(f"[{timestamp_ms}ms] Skipped: Low visibility", end="\r")
            return False

    def get_summary_statistics(self):
        if not self.left_knee_history or not self.left_foot_history:
            return None

        summary = {
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

        return summary


class FastGaitAnalyzer:
    """Simplified, fast, single-threaded gait analyzer"""

    def __init__(self, model_path, skip_frames=2, resize_scale=50, min_visibility=0.6):
        self.model_path = model_path
        self.skip_frames = skip_frames  # Process 1 of every N+1 frames
        self.resize_scale = resize_scale
        self.min_visibility = min_visibility
        self.detector = None

    def init_detector(self):
        """Initialize the pose detector"""
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
        """Resize frame for faster processing"""
        if self.resize_scale != 100:
            width = int(frame.shape[1] * self.resize_scale / 100)
            height = int(frame.shape[0] * self.resize_scale / 100)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        return frame

    def analyze_video(self, video_path):
        """Main method - simplified and fast"""

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_duration = total_frames / fps

        # Initialize detector and analyzer
        self.detector = self.init_detector()
        analyzer = GaitAnalyzer(min_visibility=self.min_visibility)

        # Print settings
        print("=" * 60)
        print("FAST GAIT ANALYZER - Simplified Version")
        print("=" * 60)
        print(f"Video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {original_duration:.2f} seconds")
        print(f"Original FPS: {fps:.2f}")
        print(
            f"Skip frames: {self.skip_frames} (processing 1/{self.skip_frames + 1} frames)"
        )
        print(f"Resize scale: {self.resize_scale}%")
        print(f"Target processing FPS: {fps / (self.skip_frames + 1):.2f}")
        print("=" * 60)

        # Process video
        frame_counter = 0
        processed_count = 0
        start_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1

            # Skip frames
            if frame_counter % (self.skip_frames + 1) != 0:
                continue

            # Preprocess
            frame = self.preprocess_frame(frame)
            timestamp_ms = int((frame_counter - 1) * 1000 / fps)

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

            # Analyze results
            if detection_result.pose_world_landmarks:
                world_landmarks = detection_result.pose_world_landmarks[0]
                analyzer.analyze_gait_cycle(world_landmarks, timestamp_ms)
                processed_count += 1

            # Progress report every 100 frames
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                progress = (frame_counter / total_frames) * 100
                print(
                    f"\nProgress: {progress:.1f}% | Processed: {processed_count} | FPS: {fps_actual:.1f} | Time: {elapsed:.1f}s",
                    end="",
                )

        # Cleanup
        cap.release()
        self.detector.close()

        # Final statistics
        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Frames processed: {processed_count}")
        print(f"Actual processing FPS: {processed_count / total_time:.2f}")
        print(f"Speedup factor: {(original_duration / total_time):.2f}x")
        print("=" * 60)

        return analyzer.get_summary_statistics()


if __name__ == "__main__":
    MODEL_PATH = "./model/pose_landmarker.task"
    VIDEO_PATH = "./dataset/healthy/man-walk.webm"

    SKIP_FRAMES = 2
    RESIZE_SCALE = 50
    MIN_VISIBILITY = 0.6

    analyzer = FastGaitAnalyzer(
        model_path=MODEL_PATH,
        skip_frames=SKIP_FRAMES,
        resize_scale=RESIZE_SCALE,
        min_visibility=MIN_VISIBILITY,
    )

    stats = analyzer.analyze_video(VIDEO_PATH)

    if stats:
        print("\n\n" + "=" * 60)
        print("                 BIOMECHANICAL GAIT DIAGNOSIS                 ")
        print("=" * 60)

        # Compute range of motion
        left_knee_rom = stats["left"]["knee_max"] - stats["left"]["knee_min"]
        right_knee_rom = stats["right"]["knee_max"] - stats["right"]["knee_min"]
        left_foot_rom = stats["left"]["foot_max"] - stats["left"]["foot_min"]
        right_foot_rom = stats["right"]["foot_max"] - stats["right"]["foot_min"]

        print("1. RANGE OF MOTION (RoM):")
        print(
            f"   Left Knee RoM:  {left_knee_rom:.1f}° | Right Knee RoM: {right_knee_rom:.1f}°"
        )
        print(
            f"   Left Foot RoM:  {left_foot_rom:.1f}° | Right Foot RoM: {right_foot_rom:.1f}°"
        )

        # Calculate normalized Symmetry Indexes
        knee_rom_si = (
            abs(left_knee_rom - right_knee_rom) / ((left_knee_rom + right_knee_rom) / 2)
        ) * 100

        foot_min_si = (
            abs(stats["left"]["foot_min"] - stats["right"]["foot_min"])
            / ((stats["left"]["foot_min"] + stats["right"]["foot_min"]) / 2)
        ) * 100

        print("\n2. NORMALIZED SYMMETRY INDEXES:")
        print(f"   Knee Flexibility Asymmetry (SI): {knee_rom_si:.1f}%")
        print(f"   Foot Drop / Extension Asymmetry (SI): {foot_min_si:.1f}%")

        svm_features = [
            left_knee_rom,
            right_knee_rom,
            stats["left"]["knee_std"],
            stats["right"]["knee_std"],
            stats["left"]["foot_min"],
            stats["right"]["foot_min"],
            knee_rom_si,
            foot_min_si,
        ]

        print("\n" + "=" * 60)
        print(f"SVM READY FEATURE VECTOR (8D Array):\n{svm_features}")
        print("=" * 60)
    else:
        print("\n Processing Failed: Not enough frames met the visibility criteria.")
        print("   Try reducing MIN_VISIBILITY to 0.5 or lower.")
