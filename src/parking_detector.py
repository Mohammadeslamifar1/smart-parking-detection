import cv2


def overlap_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0

    return (x_right - x_left) * (y_bottom - y_top)


def expand_space(space, margin=12):
    x1, y1, x2, y2 = space
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


def get_vehicle_boxes(results, model, confidence_threshold=0.30):
    vehicle_boxes = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name in ["car", "truck", "bus"] and confidence > confidence_threshold:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return vehicle_boxes


def detect_parking_spaces(
    image,
    model,
    parking_spaces,
    confidence_threshold=0.30,
    overlap_threshold=0.12,
    space_margin=12,
):
    results = model(image)

    vehicle_boxes = get_vehicle_boxes(
        results,
        model,
        confidence_threshold,
    )

    output_image = image.copy()

    occupied_count = 0
    available_count = 0

    for space in parking_spaces:
        is_occupied = False

        expanded_space = expand_space(space, margin=space_margin)
        sx1, sy1, sx2, sy2 = expanded_space
        space_area = (sx2 - sx1) * (sy2 - sy1)

        for vehicle_box in vehicle_boxes:
            area = overlap_area(expanded_space, vehicle_box)
            overlap_ratio = area / space_area

            if overlap_ratio > overlap_threshold:
                is_occupied = True
                break

        x1, y1, x2, y2 = space

        if is_occupied:
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