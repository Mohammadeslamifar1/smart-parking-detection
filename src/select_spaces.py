import cv2

image_path = "data/images/parking.jpg"
image = cv2.imread(image_path)
SAVE_PATH = "src/parking_configs/top_down_spaces.py"
# SAVE_PATH = "src/parking_configs/normal_angle_spaces.py"

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

clone = image.copy()
points = []
spaces = []

copy_mode = False
template_width = None
template_height = None


def redraw_image():
    global image
    image = clone.copy()

    for space in spaces:
        x1, y1, x2, y2 = space
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def mouse_callback(event, x, y, flags, param):
    global points, spaces, image
    global copy_mode, template_width, template_height

    if event == cv2.EVENT_LBUTTONDOWN:
        if copy_mode and template_width is not None and template_height is not None:
            x1 = x
            y1 = y
            x2 = x + template_width
            y2 = y + template_height

            spaces.append((x1, y1, x2, y2))
            print(f"Added copied parking space: ({x1}, {y1}, {x2}, {y2})")

            redraw_image()
            return

        points.append((x, y))
        print(f"Clicked point: ({x}, {y})")

        cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]

            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)

            spaces.append((left, top, right, bottom))

            template_width = right - left
            template_height = bottom - top

            print(f"Added parking space: ({left}, {top}, {right}, {bottom})")
            print(f"Template size saved: width={template_width}, height={template_height}")

            points = []
            redraw_image()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if spaces:
            removed = spaces.pop()
            print(f"Removed last parking space: {removed}")
            redraw_image()


cv2.namedWindow("Select Parking Spaces")
cv2.setMouseCallback("Select Parking Spaces", mouse_callback)

print("Instructions:")
print("First, draw one parking space using two left clicks.")
print("Press C to enable copy mode.")
print("In copy mode, one left click adds the same size rectangle.")
print("Right click removes the last selected space.")
print("Press S to save.")
print("Press Q to quit without saving.")

while True:
    cv2.imshow("Select Parking Spaces", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        if template_width is None or template_height is None:
            print("Draw one rectangle first before using copy mode.")
        else:
            copy_mode = not copy_mode
            print(f"Copy mode: {copy_mode}")

    elif key == ord("s"):
        with open(SAVE_PATH, "w") as file:
            file.write("PARKING_SPACES = [\n")
            for space in spaces:
                file.write(f"    {space},\n")
            file.write("]\n")

        print("Saved parking spaces to src/parking_spaces.py")
        break

    elif key == ord("q"):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()