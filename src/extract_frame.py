import cv2

video_path = "data/videos/parking.mp4"
image_path = "data/images/parking.jpg"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

frame_number = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read frame from the video.")

cv2.imwrite(image_path, frame)

height, width = frame.shape[:2]

print(f"Frame saved to: {image_path}")
print(f"Frame size: {width}x{height}")