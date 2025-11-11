import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import queue
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
W, H = 640, 480
FPS = 30

# Load extrinsic calibration (Left to Right)
CALIB_FILE = "extrinsic_calibration.npz"
calib = np.load(CALIB_FILE)
R = calib['R']        # 3x3 rotation
T = calib['T']        # 3x1 translation

print(f"Loaded extrinsic: R shape {R.shape}, T shape {T.shape}")

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Queues for thread-safe frame sharing
left_queue = queue.Queue(maxsize=2)
right_queue = queue.Queue(maxsize=2)

# 3D Plot setup
plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Fused 3D Pose Reconstruction (Left + Right)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_zlabel('Y (m)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0.0, 3.0)
ax.set_zlim(-1.5, 1.0)

# Default view
elev, azim = 20, -60

# -------------------------------
# RealSense Pipeline (Threaded)
# -------------------------------
def start_pipeline(device_id, q, name):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_id)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Get intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    print(f"{name} camera intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
          f"ppx={intrinsics.ppx:.2f}, ppy={intrinsics.ppy:.2f}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_frame_copy = depth_frame  # Keep reference

            if not q.full():
                q.put((color_image.copy(), depth_frame_copy, intrinsics))
            else:
                try:
                    q.get_nowait()
                except:
                    pass
                q.put((color_image.copy(), depth_frame_copy, intrinsics))

    except Exception as e:
        print(f"{name} pipeline error: {e}")
    finally:
        pipeline.stop()

# -------------------------------
# 3D Landmark Reconstruction
# -------------------------------
def get_3d_landmarks(color_image, depth_frame, intrinsics, pose_processor):
    h, w = color_image.shape[:2]
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = pose_processor.process(rgb_image)

    landmarks_3d = []
    visibility = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)

            depth = 0.0
            if 0 <= px < w and 0 <= py < h:
                depth = depth_frame.get_distance(px, py)

            if depth > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth)
            else:
                point_3d = [0, 0, 0]

            landmarks_3d.append(point_3d)
            visibility.append(lm.visibility)

    return np.array(landmarks_3d), results.pose_landmarks, np.array(visibility)

# -------------------------------
# Start Cameras
# -------------------------------
print("Detecting RealSense devices...")
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 2:
    raise RuntimeError("Need at least 2 RealSense cameras connected!")

left_serial = devices[0].get_info(rs.camera_info.serial_number)
right_serial = devices[1].get_info(rs.camera_info.serial_number)
print(f"Left camera: {left_serial}, Right camera: {right_serial}")

left_thread = threading.Thread(target=start_pipeline, args=(left_serial, left_queue, "Left"), daemon=True)
right_thread = threading.Thread(target=start_pipeline, args=(right_serial, right_queue, "Right"), daemon=True)
left_thread.start()
right_thread.start()

# Initialize MediaPipe Pose with fixed settings
pose_left = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_right = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

time.sleep(2)  # Warm-up

# -------------------------------
# Main Loop
# -------------------------------
# -------------------------------
# Main Loop with Auto-Rotation
# -------------------------------
print("Starting live 3D reconstruction... Press 'q' to quit, 'r' to reset view, 's' to toggle rotation.")

# Rotation control
auto_rotate = True
rotation_speed = 0.5  # degrees per frame (at 30 FPS → ~15°/sec; adjust as needed)
elev, azim = 20, -60  # initial view
base_azim = azim

try:
    while True:
        left_data = right_data = None
        try:
            left_data = left_queue.get_nowait()
        except:
            pass
        try:
            right_data = right_queue.get_nowait()
        except:
            pass

        if left_data is None or right_data is None:
            time.sleep(0.001)
            continue

        (color_l, depth_l, intr_l) = left_data
        (color_r, depth_r, intr_r) = right_data

        # Get 3D points from both cameras
        points_l, landmarks_l, vis_l = get_3d_landmarks(color_l, depth_l, intr_l, pose_left)
        points_r, landmarks_r, vis_r = get_3d_landmarks(color_r, depth_r, intr_r, pose_right)

        # Transform right points to left coordinate system
        valid_r = np.any(points_r != 0, axis=1)
        points_r_in_left = np.zeros_like(points_r)
        if valid_r.any():
            points_r_hom = points_r[valid_r].T
            points_r_in_left[valid_r] = (R @ points_r_hom + T).T

        # Fuse: Left preferred, Right fills missing
        points_fused = points_l.copy()
        missing_l = ~np.any(points_l != 0, axis=1)
        points_fused[missing_l] = points_r_in_left[missing_l]

        # Fuse visibility
        visibility_fused = vis_l.copy()
        visibility_fused[missing_l] = vis_r[missing_l]

        # Filter by confidence
        high_conf = visibility_fused > 0.7
        points_filtered = points_fused[high_conf]

        # Build index mapping: original -> filtered
        original_to_filtered = {}
        filtered_idx = 0
        for orig_idx in range(len(high_conf)):
            if high_conf[orig_idx]:
                original_to_filtered[orig_idx] = filtered_idx
                filtered_idx += 1

        # === AUTO ROTATE ===
        if auto_rotate:
            azim += rotation_speed
            if azim >= 300:
                azim = -60  # loop smoothly

        # Update 3D plot
        ax.clear()
        ax.set_title(f'Fused 3D Pose (Left + Right) | Rotation: {"ON" if auto_rotate else "OFF"}')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Y (m)')
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(0.0, 3.0); ax.set_zlim(-1.5, 1.0)

        if len(points_filtered) > 0:
            ax.scatter(points_filtered[:, 0], points_filtered[:, 2], -points_filtered[:, 1],
                       c='cyan', s=30, depthshade=True)

            for connection in mp_pose.POSE_CONNECTIONS:
                i, j = connection
                if i in original_to_filtered and j in original_to_filtered:
                    fi = original_to_filtered[i]
                    fj = original_to_filtered[j]
                    p1 = points_filtered[fi]
                    p2 = points_filtered[fj]
                    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], 'magenta', linewidth=2)

        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 2D Visualization
        annotated_l = color_l.copy()
        annotated_r = color_r.copy()

        if landmarks_l:
            mp_drawing.draw_landmarks(annotated_l, landmarks_l, mp_pose.POSE_CONNECTIONS)
        if landmarks_r:
            mp_drawing.draw_landmarks(annotated_r, landmarks_r, mp_pose.POSE_CONNECTIONS)

        cv2.putText(annotated_l, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_r, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack((annotated_l, annotated_r))
        cv2.imshow("Dual RealSense Pose Tracking", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            elev, azim = 20, -60
            base_azim = azim
            print("View reset.")
        elif key == ord('s'):
            auto_rotate = not auto_rotate
            print(f"Auto-rotation: {'ON' if auto_rotate else 'OFF'}")

finally:
    pose_left.close()
    pose_right.close()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)
    print("Shutdown complete.")
