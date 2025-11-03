import os
import cv2
from ultralytics import YOLO

# Input and output paths
input_path = r"C:\Users\jaysu\OneDrive\Desktop\Nexsync project\yolos\testing\video\3.mp4"
output_path = r"C:\Users\jaysu\OneDrive\Desktop\Nexsync project\yolos\testing\video\out\3_outv2.mp4"

# Load video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError(f"Error opening video file: {input_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()
H, W, _ = frame.shape

# Define video writer
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# Load YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'best.pt')
model = YOLO(model_path)

threshold = 0.5

while ret:
    # Run YOLO inference
    results = model(frame)[0]

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Saved output to: {output_path}")
