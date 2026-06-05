import cv2

from parking_configs.top_down_spaces import PARKING_SPACES
from parking_detector_pixels import detect_parking_spaces_by_pixels


video_path = "data/videos/parking.mp4"
output_path = "outputs/parking_video_pixels_result.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    success, frame_bgr = cap.read()

    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    result_rgb, total_count, available_count, occupied_count = detect_parking_spaces_by_pixels(
        image=frame_rgb,
        parking_spaces=PARKING_SPACES,
        pixel_threshold=500,
    )

    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    out.write(result_bgr)

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()

print("Video processing complete.")
print(f"Total spaces: {total_count}")
print(f"Available spaces: {available_count}")
print(f"Occupied spaces: {occupied_count}")
print(f"Result saved to {output_path}")