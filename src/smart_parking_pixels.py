import cv2

from parking_spaces import PARKING_SPACES
from parking_detector_pixels import detect_parking_spaces_by_pixels


image_path = "data/images/parking.png"
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

result_rgb, total_count, available_count, occupied_count = detect_parking_spaces_by_pixels(
    image=image_rgb,
    parking_spaces=PARKING_SPACES,
    pixel_threshold=150,
)

result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("outputs/smart_parking_pixels_result.jpg", result_bgr)

print("Pixel based parking detection complete.")
print(f"Total spaces: {total_count}")
print(f"Available spaces: {available_count}")
print(f"Occupied spaces: {occupied_count}")
print("Result saved to outputs/smart_parking_pixels_result.jpg")