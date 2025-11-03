from ultralytics import YOLO
import cv2
import os

frame = cv2.imread(r"C:\Users\jaysu\OneDrive\Desktop\Nexsync project\yolos\testing\images\41.jpg")

model_path = r'.\runs\detect\train3\weights\best.pt' 
model = YOLO(model_path)

frame_resized = cv2.resize(frame, (640, 640))
results = model(frame_resized)[0] 
threshold = 0.25

h_ratio = frame.shape[0] / 640
w_ratio = frame.shape[1] / 640

for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()  
    score = box.conf[0].item()              
    class_id = int(box.cls[0].item())       

    if score > threshold:
        x1 = int(x1 * w_ratio)
        x2 = int(x2 * w_ratio)
        y1 = int(y1 * h_ratio)
        y2 = int(y2 * h_ratio)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, results.names[class_id].upper(), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

cv2.imwrite(r"C:\Users\jaysu\OneDrive\Desktop\Nexsync project\yolos\testing\images\out\41_out.jpg",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
