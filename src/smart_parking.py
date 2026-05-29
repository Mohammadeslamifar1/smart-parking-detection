import cv2
from ultralytics import YOLO

from parking_spaces import PARKING_SPACES
from parking_detector import detect_parking_spaces


model = YOLO("yolov8n.pt")

image_path = "data/images/parking.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result_image, total_count, available_count, occupied_count = detect_parking_spaces(
    image=image_rgb,
    model=model,
    parking_spaces=PARKING_SPACES,
    confidence_threshold=0.30,
    overlap_threshold=0.12,
)

result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("outputs/smart_parking_result.jpg", result_bgr)

print("Smart parking detection complete.")
print(f"Total spaces: {total_count}")
print(f"Available spaces: {available_count}")
print(f"Occupied spaces: {occupied_count}")
print("Result saved to outputs/smart_parking_result.jpg")