import cv2
import mediapipe as mp
import json

#video_path = "soccer_action.mp4" # shoumen action
#output_video_path = "pose_output.mp4"
#output_json_path = "pose_data.json"

action = "dianqiu3" #"dianqiu1"

video_path = action + ".mp4"
output_video_path = action + "_output.mp4"
output_json_path = action + ".json"


OUTPUT_WIDTH, OUTPUT_HEIGHT = 640, 480

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

 cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video file.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

pose_data = []  # list of frame data

with mp_pose.Pose(static_image_mode=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_landmarks = []

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        # Save landmarks to the list
        pose_data.append({
            "frame": frame_idx,
            "landmarks": frame_landmarks
        })

        out.write(frame)
        frame_idx += 1

cap.release()
out.release()

with open(output_json_path, 'w') as f:
    json.dump(pose_data, f, indent=2)

import json
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

def load_pose_sequence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    sequence = []
    invalid_frames = 0
    for frame_data in data:
        landmarks = frame_data.get("landmarks", [])
        if len(landmarks) != 33:
            invalid_frames += 1
            continue
            
        frame = []
        for lm in landmarks:
            frame.extend([lm["x"], lm["y"], lm["z"]])
        sequence.append(np.array(frame))  # shape: (99,)
    print(f"Skipped {invalid_frames} invalid frames")
    
    return sequence

def normalize_pose_sequence(sequence):
    normalized = []
    for frame in sequence:
        #midpoint between left hip (23) and right hip (24)
        mid_hip = (frame[23] + frame[24]) / 2.0

        # distance between left shoulder (11) and right shoulder (12)
        shoulder_dist = np.linalg.norm(frame[11] - frame[12])
        scale = shoulder_dist if shoulder_dist > 1e-5 else 1.0  # prevent divide-by-zero

        # Normalize: center and scale
        norm_frame = (frame - mid_hip) / scale
        normalized.append(norm_frame)
    return normalized


def compute_dtw_similarity(seq1, seq2):
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return distance, path

# Load and flatten pose sequences
json1_path = 'dianqiu2.json'  # expert
json2_path = 'dianqiu3.json'  # trainee

seq1 = load_pose_sequence(json1_path)
seq2 = load_pose_sequence(json2_path)

# After loading sequences
seq1 = normalize_pose_sequence(seq1)
seq2 = normalize_pose_sequence(seq2)

#flat_seq1 = flatten_sequence(seq1)
#flat_seq2 = flatten_sequence(seq2)

# Compute DTW similarity
dtw_distance, alignment_path = compute_dtw_similarity(seq1, seq2)

print(f"DTW distance: {dtw_distance}")
print(f"Alignment path (first 10 pairs): {alignment_path[:10]}")

import json
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

def load_pose_sequence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    sequence = []
    invalid_frames = 0
    for frame_data in data:
        landmarks = frame_data.get("landmarks", [])
        if len(landmarks) != 33:
            invalid_frames += 1
            continue
            
        frame = []
        for lm in landmarks:
            frame.extend([lm["x"], lm["y"], lm["z"]])
        sequence.append(np.array(frame))  # shape: (99,)
    print(f"Skipped {invalid_frames} invalid frames")
    
    return sequence

def normalize_pose_sequence(sequence):
    normalized = []
    for frame in sequence:
        #midpoint between left hip (23) and right hip (24)
        mid_hip = (frame[23] + frame[24]) / 2.0

        # distance between left shoulder (11) and right shoulder (12)
        shoulder_dist = np.linalg.norm(frame[11] - frame[12])
        scale = shoulder_dist if shoulder_dist > 1e-5 else 1.0  # prevent divide-by-zero

        # Normalize: center and scale
        norm_frame = (frame - mid_hip) / scale
        normalized.append(norm_frame)
    return normalized


def compute_dtw_similarity(seq1, seq2):
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return distance, path

# Load and flatten pose sequences
json1_path = 'dianqiu2.json'  # expert
json2_path = 'dianqiu3.json'  # trainee

seq1 = load_pose_sequence(json1_path)
seq2 = load_pose_sequence(json2_path)

# After loading sequences
seq1 = normalize_pose_sequence(seq1)
seq2 = normalize_pose_sequence(seq2)

#flat_seq1 = flatten_sequence(seq1)
#flat_seq2 = flatten_sequence(seq2)

# Compute DTW similarity
dtw_distance, alignment_path = compute_dtw_similarity(seq1, seq2)

print(f"DTW distance: {dtw_distance}")
print(f"Alignment path (first 10 pairs): {alignment_path[:10]}")

## full-body overlay
import json
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def load_poses(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    sequence = []
    invalid_frames = 0
    for frame_data in data:
        landmarks = frame_data.get("landmarks", [])
        if len(landmarks) != 33:
            invalid_frames += 1
            continue
            
        frame = []
        for lm in landmarks:
            frame.extend([lm["x"], lm["y"]])
        sequence.append(np.array(frame))  # shape: (99,)
    print(f"Skipped {invalid_frames} invalid frames")
    
    return sequence


def normalize_pose(pose):
    pose = pose - np.mean(pose, axis=0)  # Center
    scale = np.linalg.norm(np.max(pose, axis=0) - np.min(pose, axis=0))
    return pose / (scale + 1e-8)

def normalize_frames(frames):
    return [normalize_pose(f) for f in frames]

def compute_dtw_alignment(seq1, seq2):
    # Flatten each frame into a 1D array
    seq1_flat = [f.flatten() for f in seq1]
    seq2_flat = [f.flatten() for f in seq2]

    distance, path = fastdtw(seq1_flat, seq2_flat, dist=euclidean)

    # Unzip the path into separate index lists
    idx1, idx2 = zip(*path)
    return list(idx1), list(idx2)

def plot_aligned_poses(seq1, seq2, idx1, idx2, connections, step=5):
    for i in range(0, len(idx1), step):
        frame1 = seq1[idx1[i]]
        frame2 = seq2[idx2[i]]
        
        # Reshape if input is flattened
        pose1 = frame1.reshape(-1, 2)
        pose2 = frame2.reshape(-1, 2)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f'Aligned Frame {i}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.invert_yaxis()
        
        # Plot first pose in red
        for connection in connections:
            x1, y1 = pose1[connection[0]]
            x2, y2 = pose1[connection[1]]
            ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.7)
        
        # Plot second pose in blue
        for connection in connections:
            x1, y1 = pose2[connection[0]]
            x2, y2 = pose2[connection[1]]
            ax.plot([x1, x2], [y1, y2], 'b--', alpha=0.7)

        # Optional: overlay keypoints
        ax.scatter(pose1[:, 0], pose1[:, 1], c='red', s=10)
        ax.scatter(pose2[:, 0], pose2[:, 1], c='blue', s=10)

        plt.show()


# ---- Pose connections (MediaPipe style) ----
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23,25), (24,26), (25,27), (26,28)
]

# ---- Run the process ----
file1 = 'dianqiu1.json'
file2 = 'dianqiu3.json'

poses1 = normalize_frames(load_poses(file1))
poses2 = normalize_frames(load_poses(file2))

idx1, idx2 = compute_dtw_alignment(poses1, poses2)

plot_aligned_poses(poses1, poses2, idx1, idx2, POSE_CONNECTIONS, step=5)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def save_aligned_poses_video(seq1, seq2, idx1, idx2, connections, out_path='aligned_poses.mp4', step=1, fps=10):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.invert_yaxis()
    ax.set_title("Aligned Motion Overlays")

    red_lines = [ax.plot([], [], 'r-', alpha=0.7)[0] for _ in connections]
    blue_lines = [ax.plot([], [], 'b--', alpha=0.7)[0] for _ in connections]
    red_dots = ax.scatter([], [], c='red', s=10)
    blue_dots = ax.scatter([], [], c='blue', s=10)

    def init():
        for line in red_lines + blue_lines:
            line.set_data([], [])
        red_dots.set_offsets(np.empty((0, 2)))
        blue_dots.set_offsets(np.empty((0, 2)))
        return red_lines + blue_lines + [red_dots, blue_dots]

    def update(i):
        frame1 = seq1[idx1[i]]
        frame2 = seq2[idx2[i]]
    
        # Ensure pose is reshaped to (num_joints, 2)
        pose1 = np.array(frame1).reshape(-1, 2)
        pose2 = np.array(frame2).reshape(-1, 2)
    
        for j, (p1, p2) in enumerate(connections):
            red_lines[j].set_data([pose1[p1][0], pose1[p2][0]], [pose1[p1][1], pose1[p2][1]])
            blue_lines[j].set_data([pose2[p1][0], pose2[p2][0]], [pose2[p1][1], pose2[p2][1]])
    
        red_dots.set_offsets(pose1)
        blue_dots.set_offsets(pose2)
    
        return red_lines + blue_lines + [red_dots, blue_dots]


    ani = animation.FuncAnimation(
        fig, update, frames=range(0, len(idx1), step),
        init_func=init, blit=True, interval=1000/fps
    )

    # Save video using FFMpeg
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='DTW Alignment'), bitrate=1800)
    ani.save(out_path, writer=writer)
    plt.close()

save_aligned_poses_video(poses1, poses2, idx1, idx2, POSE_CONNECTIONS, out_path="aligned_overlay.mp4", step=2, fps=10)
