import cv2
import os
import numpy as np
import mediapipe as mp
import torch

# Parameters
DATASET_PATH = "C:\\Users\\ADITYA RAJ\\OneDrive\\Desktop\\ISL Dataset"
IMAGE_SIZE = (64, 64)
FRAMES_PER_VIDEO = 30
NUM_LANDMARKS = 21 * 3  # 21 hand landmarks, each with (x, y, z) coordinates

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)  # Only track one hand

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"⚠ Warning: Could not open {video_path}")
        return np.zeros((FRAMES_PER_VIDEO, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)

    while len(frames) < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMAGE_SIZE)
        frames.append(frame)

    cap.release()

    # Ensure exactly FRAMES_PER_VIDEO by padding with black frames if needed
    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8))

    return np.array(frames)

def extract_hand_landmarks(image):
    """
    Extracts 21 hand landmarks (x, y, z). If no hand is detected, returns 63 zeros.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    landmarks = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Take only the first detected hand
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

    # If no hand detected, return zeros
    if len(landmarks) == 0:
        return np.zeros(63, dtype=np.float32)

    return np.array(landmarks, dtype=np.float32).flatten()  # Ensure shape (63,)

# Load dataset
X_data, y_labels = [], []
labels_dict = {}

for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):  # Sort for consistency
    class_path = os.path.join(DATASET_PATH, class_name)
    labels_dict[idx] = class_name

    for video_file in sorted(os.listdir(class_path)):  # Sort for consistency
        video_path = os.path.join(class_path, video_file)
        
        frames = extract_frames(video_path)

        # Extract landmarks for all frames
        landmarks_seq = np.array([extract_hand_landmarks(frame) for frame in frames])

        # Debugging output
        print(f"Processed video: {video_file}, Landmarks shape: {landmarks_seq.shape}")

        # Ensure correct shape (30, 63)
        if landmarks_seq.shape == (FRAMES_PER_VIDEO, NUM_LANDMARKS):
            X_data.append(landmarks_seq)
            y_labels.append(idx)
        else:
            print(f"⚠ Skipping {video_path} due to shape mismatch: {landmarks_seq.shape}")

# Convert to NumPy arrays
X_data = np.stack(X_data)  # Ensures uniform shape
y_labels = np.array(y_labels, dtype=np.int64)

# Save data as Torch tensors
torch.save(torch.tensor(X_data), "X_data.pt")
torch.save(torch.tensor(y_labels), "y_labels.pt")
torch.save(labels_dict, "labels_dict.pt")

print(f"✅ Data saved successfully! X_data shape: {X_data.shape}, y_labels shape: {y_labels.shape}")
print(f"Training data shape: {X_data.shape}")  # Check shape
