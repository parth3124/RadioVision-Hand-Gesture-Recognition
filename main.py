import cv2
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
from collections import deque
import numpy as np

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# LOAD LABELS
# -------------------------------
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Loaded classes:", class_names)

# -------------------------------
# LOAD MODEL
# -------------------------------
num_classes = len(class_names)

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("gesture_model.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# TRANSFORMS (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------------
# PREDICTION SMOOTHING
# -------------------------------
pred_queue = deque(maxlen=5)

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    # ROI coordinates
    x1, y1 = 350, 100
    x2, y2 = 600, 350

    p_time = 0

    print("🚀 Started Real-Time Gesture Recognition")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to access webcam.")
            break

        img = cv2.flip(img, 1)

        # Draw ROI box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Place hand in box", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        # Crop ROI
        roi = img[y1:y2, x1:x2]

        # -------------------------------
        # PREPROCESSING IMPROVEMENTS
        # -------------------------------

        # Lighting normalization
        roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)

        # Noise reduction
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        # Check if ROI is valid
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        # Convert to RGB → PIL
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        # Transform
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # -------------------------------
        # MODEL PREDICTION
        # -------------------------------
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, 1)

            # Top-2 gap (for better filtering)
            top2 = torch.topk(probs, 2)
            gap = (top2.values[0][0] - top2.values[0][1]).item()

        confidence_score = confidence.item() * 100

        # -------------------------------
        # SMOOTHING
        # -------------------------------
        pred_queue.append(predicted.item())

        final_pred = max(set(pred_queue), key=pred_queue.count)
        predicted_class = class_names[final_pred]

        # -------------------------------
        # DECISION LOGIC
        # -------------------------------
        if confidence_score > 70 and gap > 0.2:
            text = f"{predicted_class} ({confidence_score:.1f}%)"
            color = (0, 255, 0)
        else:
            text = "Uncertain..."
            color = (0, 0, 255)

        # -------------------------------
        # DISPLAY
        # -------------------------------
        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Confidence bar (for demo)
        bar_width = int(confidence_score * 2)
        cv2.rectangle(img, (20, 100), (20 + bar_width, 120), color, -1)

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 150),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Show windows
        cv2.imshow("Hand Gesture Recognition", img)
        cv2.imshow("ROI", roi)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()