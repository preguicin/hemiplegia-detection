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
            return

        # Lower Body Landmarks
        left_hip = world_landmarks[23]
        left_knee = world_landmarks[25]
        left_ankle = world_landmarks[27]
        left_foot = world_landmarks[31]  # Foot Index (Toes)

        right_hip = world_landmarks[24]
        right_knee = world_landmarks[26]
        right_ankle = world_landmarks[28]
        right_foot = world_landmarks[32]  # Foot Index (Toes)

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
            # 1. Calculate Standard Knee Flexion Angles (Hip -> Knee -> Ankle)
            left_knee_angle = calculate_joint_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_joint_angle(right_hip, right_knee, right_ankle)

            # 2. Calculate Foot-to-Shin Articular Angles (Knee -> Ankle -> Toes)
            left_foot_angle = calculate_joint_angle(left_knee, left_ankle, left_foot)
            right_foot_angle = calculate_joint_angle(
                right_knee, right_ankle, right_foot
            )

            # Commit calculations to historical feature arrays
            self.left_knee_history.append(left_knee_angle)
            self.right_knee_history.append(right_knee_angle)
            self.left_foot_history.append(left_foot_angle)
            self.right_foot_history.append(right_foot_angle)

            print(
                f"[{timestamp_ms}ms] KNEES L: {left_knee_angle:.1f}° R: {right_knee_angle:.1f}° | FEET L: {left_foot_angle:.1f}° R: {right_foot_angle:.1f}°",
                end="\r",
            )
        else:
            print(
                f"[{timestamp_ms}ms] Processing Skipped: Landmarks obscured.", end="\r"
            )

    def get_summary_statistics(self):
        if not self.left_knee_history or not self.left_foot_history:
            return None

        summary = {
            "left": {
                "knee_min": np.min(self.left_knee_history),
                "knee_max": np.max(self.left_knee_history),
                "knee_avg": np.mean(self.left_knee_history),
                "foot_min": np.min(self.left_foot_history),
                "foot_max": np.max(self.left_foot_history),
                "foot_avg": np.mean(self.left_foot_history),
            },
            "right": {
                "knee_min": np.min(self.right_knee_history),
                "knee_max": np.max(self.right_knee_history),
                "knee_avg": np.mean(self.right_knee_history),
                "foot_min": np.min(self.right_foot_history),
                "foot_max": np.max(self.right_foot_history),
                "foot_avg": np.mean(self.right_foot_history),
            },
        }

        return summary


base_options = python.BaseOptions(model_asset_path="./src/pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
)
detector = vision.PoseLandmarker.create_from_options(options)

# Lower body & hips connections list
POSE_CONNECTIONS = [
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
]

video_path = "./dataset/hemiplegia/gait.webm"
cap = cv2.VideoCapture(video_path)
analyzer = GaitAnalyzer(min_visibility=0.6)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_height, frame_width, _ = frame.shape
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        world_landmarks = (
            detection_result.pose_world_landmarks[0]
            if detection_result.pose_world_landmarks
            else None
        )

        analyzer.analyze_gait_cycle(world_landmarks, frame_timestamp_ms)

        pixel_coords = {}
        for idx, landmark in enumerate(landmarks):
            if landmark.visibility > 0.5:
                cx, cy = int(landmark.x * frame_width), int(landmark.y * frame_height)
                pixel_coords[idx] = (cx, cy)

                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in pixel_coords and end_idx in pixel_coords:
                cv2.line(
                    frame,
                    pixel_coords[start_idx],
                    pixel_coords[end_idx],
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("MediaPipe Tasks - Annotated Body Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stats = analyzer.get_summary_statistics()

if stats:
    print("\n\n--- Final Extraction Results ---")
    print(f"Gait Stats: {stats!r}")
else:
    print("\nProcessing Failed: Not enough frames met the paired visibility criteria.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()
