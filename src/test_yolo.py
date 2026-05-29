from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "data/images/parkingTEST.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

results = model(image)

annotated_image = results[0].plot()

cv2.imwrite("outputs/yolo_test_result.jpg", annotated_image)

print("Detection complete. Result saved to outputs/yolo_test_result.jpg")