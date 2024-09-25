from ultralytics import YOLO
import cv2
import numpy as np
import random
from deep_sort_realtime.deepsort_tracker import DeepSort

# Open the video file
cap = cv2.VideoCapture("supermarket.mp4")
cap.set(3, 1280)

# Get video dimensions (for scaling the 2D space)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a blank canvas for the 2D tracks
canvas_height, canvas_width = 800, 800  # You can adjust this
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Load the YOLOv8 model
model = YOLO('yolo/yolov8l.pt')

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30)

# Dictionary to store the history of positions (footprints) for each track_id
footprints = {}

# Dictionary to store a unique color for each track_id
colors = {}

# Function to generate random colors
def get_color(track_id):
    if track_id not in colors:
        colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
    return colors[track_id]

# Function to scale the coordinates to the canvas size
def scale_coordinates(cx, cy, img_width, img_height, canvas_width, canvas_height):
    scaled_x = int(cx * canvas_width / img_width)
    scaled_y = int(cy * canvas_height / img_height)
    return scaled_x, scaled_y

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

    # Iterate over the tracked objects and draw the bounding boxes and ID
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = map(int, track.to_ltwh())  # Bounding box coordinates
        track_id = int(track.track_id)  # Track ID from DeepSORT

        # Get the unique color for this track_id
        color = get_color(track_id)

        # Calculate the centroid of the bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Add the current centroid to the footprint history
        if track_id not in footprints:
            footprints[track_id] = []  # Initialize a new list for the person

        footprints[track_id].append((cx, cy))  # Append the new position

        # Scale the centroid coordinates to the canvas size
        scaled_cx, scaled_cy = scale_coordinates(cx, cy, frame_width, frame_height, canvas_width, canvas_height)

        # Draw the footprint trail (line connecting the past centroids) on the 2D canvas
        for j in range(1, len(footprints[track_id])):
            if footprints[track_id][j - 1] is None or footprints[track_id][j] is None:
                continue
            # Scale previous and current centroids
            prev_scaled_cx, prev_scaled_cy = scale_coordinates(footprints[track_id][j - 1][0], footprints[track_id][j - 1][1], frame_width, frame_height, canvas_width, canvas_height)
            curr_scaled_cx, curr_scaled_cy = scale_coordinates(footprints[track_id][j][0], footprints[track_id][j][1], frame_width, frame_height, canvas_width, canvas_height)
            
            # Draw a line between the previous and current points on the 2D canvas
            cv2.line(canvas, (prev_scaled_cx, prev_scaled_cy), (curr_scaled_cx, curr_scaled_cy), color, 2)
    
    # Show the 2D canvas with the tracked paths
    cv2.imshow("2D Tracks", canvas)

    # Also show the original video frame (optional)
    cv2.imshow("Original Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
