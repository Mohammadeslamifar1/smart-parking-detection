import cv2
import numpy as np


def detect_parking_spaces_by_pixels(
    image,
    parking_spaces,
    pixel_threshold=900,
):
    output_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)

    edges = cv2.Canny(blur, 50, 150)

    occupied_count = 0
    available_count = 0

    for space in parking_spaces:
        x1, y1, x2, y2 = space

        crop = edges[y1:y2, x1:x2]

        non_zero_pixels = cv2.countNonZero(crop)

        if non_zero_pixels > pixel_threshold:
            color = (255, 0, 0)
            occupied_count += 1
        else:
            color = (0, 255, 0)
            available_count += 1

        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

    total_count = available_count + occupied_count

    summary = f"Total: {total_count} | Free: {available_count} | Used: {occupied_count}"

    cv2.putText(
        output_image,
        summary,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return output_image, total_count, available_count, occupied_count