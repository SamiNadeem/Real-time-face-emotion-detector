import cv2
from fer import FER
import numpy as np

print("=" * 50)
print("FER Diagnostic Test")
print("=" * 50)

# Step 1: Camera test
print("\n[1] Opening camera...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read camera frame")
    exit()

print("OK - Frame shape:", frame.shape)

# Step 2: FER model load
print("\n[2] Loading FER model...")
detector = FER(mtcnn=False)
print("OK - FER loaded")

# Step 3: Run detection
print("\n[3] Running detection on camera frame...")
result = detector.detect_emotions(frame)
print("Raw result:", result)

# Step 4: Interpret result
print("\n[4] Result summary:")
if not result:
    print("NO FACES DETECTED")
    print("  - Make sure face is well lit")
    print("  - Sit closer to camera (40-60cm)")
    print("  - Look directly at camera")
else:
    for i, face in enumerate(result):
        print("Face", i+1, "box:", face["box"])
        print("Face", i+1, "emotions:", face["emotions"])
        top = max(face["emotions"], key=face["emotions"].get)
        print("Top emotion:", top, "-", round(face["emotions"][top]*100), "%")

print("\n" + "=" * 50)
print("Test complete")
print("=" * 50)
