import cv2
import torch
import numpy as np
import mediapipe as mp
from train import SignLanguageLSTM



# Load model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 63  # Each frame has 63 features (21 landmarks × 3 coordinates)
HIDDEN_SIZE = 128  # Can be adjusted based on your model

model = SignLanguageLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=11).to(device)
labels_dict = torch.load("labels_dict.pt")# Adjust `num_classes` as needed

# Load only weights (safe)
model.load_state_dict(torch.load("train.pth", weights_only=True))

model.eval()  # Set to evaluation mode
# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    landmarks = []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Take only the first detected hand
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

    # Ensure exactly 63 features (21 landmarks × 3)
    if len(landmarks) < 21:
        landmarks += [[0.0, 0.0, 0.0]] * (21 - len(landmarks))  # Pad missing landmarks

    return np.array(landmarks).flatten()  # Shape will always be (63,)

# Start webcam
cap = cv2.VideoCapture(0)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    landmarks = extract_hand_landmarks(frame)

    if len(frames) < 30:
        frames.append(landmarks)
    else:
        frames.pop(0)
        frames.append(landmarks)

        input_data = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(input_data)
            class_id = torch.argmax(prediction).item()
            class_name = labels_dict[class_id]

        cv2.putText(frame, f"Prediction: {class_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
