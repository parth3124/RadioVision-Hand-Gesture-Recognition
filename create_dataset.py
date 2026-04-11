import cv2
import os

# -------------------------------
# CONFIG
# -------------------------------
SAVE_PATH = "dataset"
IMG_SIZE = 224
MAX_IMAGES_PER_CLASS = 300

GESTURES = {
    "0": "palm",
    "1": "fist",
    "2": "peace",
    "3": "thumbs_up",
    "4": "ok"
}

# -------------------------------
# CREATE FOLDERS
# -------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)

for gesture in GESTURES.values():
    os.makedirs(os.path.join(SAVE_PATH, gesture), exist_ok=True)

# -------------------------------
# COUNT EXISTING IMAGES
# -------------------------------
image_counts = {}

for gesture in GESTURES.values():
    folder = os.path.join(SAVE_PATH, gesture)
    image_counts[gesture] = len(os.listdir(folder))

current_label = None

# -------------------------------
# START WEBCAM
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Webcam not working")
    exit()

print("\n=== DATA COLLECTION ===")
print("Press 0-4 to select gesture")
print("Press 's' to save")
print("Press 'q' to quit\n")

# ROI box (VERY IMPORTANT → same as main.py)
x1, y1 = 350, 100
x2, y2 = 600, 350

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop ROI
    roi = frame[y1:y2, x1:x2]

    # Resize (VERY IMPORTANT → matches training)
    hand_crop = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    # Show current class
    if current_label:
        cv2.putText(frame, f"Class: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Crop", hand_crop)

    key = cv2.waitKey(1) & 0xFF

    # Select class
    if chr(key) in GESTURES:
        current_label = GESTURES[chr(key)]
        print("Selected:", current_label)

    # Save image
    elif key == ord('s'):
        if current_label is None:
            print("Select class first!")
        else:
            save_dir = os.path.join(SAVE_PATH, current_label)
            filename = f"{current_label}_{image_counts[current_label]}.jpg"
            path = os.path.join(save_dir, filename)

            cv2.imwrite(path, hand_crop)
            image_counts[current_label] += 1

            print("Saved:", path)

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()