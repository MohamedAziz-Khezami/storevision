from ultralytics import YOLO
import cv2
import numpy as np
import random
from deep_sort_realtime.deepsort_tracker import DeepSort

# Open the video file
cap = cv2.VideoCapture("supermarket.mp4")
cap.set(3, 1280)

# Load the YOLOv8 model
model = YOLO('yolo/yolov8l.pt')

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30)

# Dictionary to store the history of positions (footprints) for each track_id
footprints = {}

# Dictionary to store a unique color for each track_id
colors = {}

# Variable to track the maximum number of people detected
max_people = 0

# Function to generate random colors
def get_color(track_id):
    if track_id not in colors:
        colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
    return colors[track_id]

while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    # Run YOLOv8 on the image (stream=True to process in real-time)
    results = model(img, stream=True)
    
    # Initialize a list to hold detections for tracking
    detections = []

    # Iterate through each result in the stream
    for result in results:
        # Extract bounding boxes and class probabilities (tensor format)
        boxes = result.boxes  # Access to xyxy, conf, cls
        
        # Convert boxes to a format compatible with DeepSORT tracker
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class label (e.g., person)
            
            # Filter by person class (assuming class 0 is 'person')
            if cls == 0:
                detections.append(([x1, y1, x2, y2], conf))

    # Update tracker with detections
    tracked_objects = tracker.update_tracks(detections, frame=img)

    # Count the number of confirmed tracks (people being tracked)
    num_people = sum(1 for track in tracked_objects if track.is_confirmed())

    # Update the maximum number of people tracked
    if num_people > max_people:
        max_people = num_people

    # Iterate over the tracked objects and draw the bounding boxes and ID
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = map(int, track.to_ltwh())  # Bounding box coordinates
        track_id = int(track.track_id)  # Track ID from DeepSORT

        # Get the unique color for this track_id
        color = get_color(track_id)

        # Draw the bounding box with the unique color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate the centroid of the bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Add the current centroid to the footprint history
        if track_id not in footprints:
            footprints[track_id] = []  # Initialize a new list for the person

        footprints[track_id].append((cx, cy))  # Append the new position

        # Draw the footprint trail (line connecting the past centroids) with the unique color
        for j in range(1, len(footprints[track_id])):
            if footprints[track_id][j - 1] is None or footprints[track_id][j] is None:
                continue
            # Draw a line between the previous and current points
            cv2.line(img, footprints[track_id][j - 1], footprints[track_id][j], color, 2)
    
    # Display the maximum number of people tracked on the screen
    cv2.putText(img, f'Max People: {max_people}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image with bounding boxes and footprints
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
