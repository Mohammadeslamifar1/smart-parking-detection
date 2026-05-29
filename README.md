# Smart Parking Space Detection

A computer vision project that detects available and occupied parking spaces from a parking lot image using YOLO, OpenCV, and Streamlit.

## Project Overview

This project uses a pretrained YOLO object detection model to detect vehicles in a parking lot image. The detected vehicle boxes are compared with manually selected parking space regions. Each parking space is then classified as either available or occupied.

The project includes both a Python script and a Streamlit web application.

## Demo

### Streamlit App

![Streamlit App](screenshots/streamlit_app(1)(2).jpg)

### Detection Result

![Detection Result](screenshots/detection_result.png)

## Features

1. Detects vehicles such as cars, trucks, and buses
2. Classifies parking spaces as available or occupied
3. Displays green boxes for available parking spaces
4. Displays red boxes for occupied parking spaces
5. Shows total, available, and occupied parking counts
6. Provides a Streamlit web interface
7. Allows users to upload a parking lot image
8. Allows users to download the result image
9. Includes a manual parking space selection tool

## Technologies Used

Python  
OpenCV  
YOLO  
Ultralytics  
Streamlit  
NumPy  
Pillow  


## Project Structure

```text
smart parking detection/
    app.py
    requirements.txt
    README.md
    data/
        images/
            parking.jpg
    outputs/
        smart_parking_result.jpg
        streamlit_result.jpg
    src/
        __init__.py
        parking_spaces.py
        parking_detector.py
        select_spaces.py
        smart_parking.py
        test_yolo.py
```

## How It Works

1. The user provides a parking lot image.
2. YOLO detects vehicles in the image.
3. The app loads predefined parking space coordinates from `parking_spaces.py`.
4. Each parking space is compared with detected vehicle boxes.
5. If a vehicle overlaps with a parking space enough, the space is marked as occupied.
6. If there is not enough overlap, the space is marked as available.
7. The final result is displayed with green and red boxes.
8. The app shows the total number of spaces, available spaces, and occupied spaces.

## Color Legend

Green boxes mean available parking spaces.  
Red boxes mean occupied parking spaces.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/smart parking detection.git
cd smart parking detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

### 6. Run the Python detection script

```bash
python src\smart_parking.py
```

## How to Select Parking Spaces

The project includes a manual parking space selection tool.

Run:

```bash
python src\select_spaces.py
```

Controls:

```text
Left click two corners of a parking space to add it
Right click to remove the last selected parking space
Press S to save selected spaces
Press Q to quit without saving
```

The selected parking spaces are saved automatically in:

```text
src/parking_spaces.py
```

## Output

The system produces an image with:

Total number of parking spaces  
Number of available spaces  
Number of occupied spaces  
Green boxes for available spaces  
Red boxes for occupied spaces  

The output image is saved in the `outputs` folder.


## Streamlit Web App

The Streamlit app allows the user to:

Upload a parking lot image  
View the original image  
View the detection result  
Adjust detection thresholds  
See parking space statistics  
Download the result image  

## Limitations

This version works best when the uploaded image has the same camera angle and image size used when selecting the parking spaces.

If the camera angle changes, the parking spaces should be selected again.

The system depends on YOLO vehicle detection, so very small, hidden, or unclear vehicles may not always be detected correctly.

## Future Improvements

Add video support  
Add real time webcam support  
Allow parking space selection directly inside the Streamlit app  
Train a custom model for parking space detection  
Improve support for different camera angles  
Deploy the app online  
Add dashboard analytics  
Add support for multiple parking lot views  

## What I Learned

Through this project, I practiced:

Computer vision basics  
Object detection using YOLO  
Image processing with OpenCV  
Manual annotation of parking spaces  
Bounding box overlap logic  
Threshold tuning  
Building a Streamlit web app  
Organizing a Python project  
Preparing a project for GitHub and portfolio use  

## Author

MOHAMMAD ESLAMI FAR