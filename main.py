from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort

# Open the video file
cap = cv2.VideoCapture("supermarket.mp4")
cap.set(3, 1280)

# Load the YOLOv8 model (runs on CPU by default if no 'device' argument is provided)
model = YOLO('yolo/yolov8l.pt')  # Default device is CPU

# Initialize the SORT tracker
tracker = Sort(max_age=90000000000 )

# Dictionary to store the history of positions (footprints) for each track_id
footprints = {}

# Dictionary to keep track of missing frames for each track_id
missing_frames = {}

# Maximum number of frames a person can be undetected before removing their track
max_missing_frames = 9000000000000

while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    # Run YOLOv8 on the image (stream=True to process in real-time)
    results = model(img, stream=True)
    
    # Initialize a list to hold detections for tracking
    detections = []
    
    # List of current frame's track_ids
    current_track_ids = []

    # Iterate through each result in the stream
    for result in results:
        # Extract bounding boxes and class probabilities (tensor format)
        boxes = result.boxes  # Access to xyxy, conf, cls
        
        # Convert boxes to a format compatible with SORT tracker
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class label (e.g., person)
            
            # Filter by person class (assuming class 0 is 'person')
            if cls == 0:
                detections.append([x1, y1, x2, y2, conf])

    # Convert detections to numpy array for SORT
    detections_np = np.array(detections)
    
    # Track the detected objects
    if len(detections_np) > 0:
        tracked_objects = tracker.update(detections_np)
        
        # Process tracked objects
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            current_track_ids.append(track_id)
            
            # Calculate the centroid of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Add the current centroid to the footprint history
            if track_id not in footprints:
                footprints[track_id] = []  # Initialize a new list for the person
                missing_frames[track_id] = 0  # Reset the missing frame count
            
            footprints[track_id].append((cx, cy))  # Append the new position
            missing_frames[track_id] = 0  # Reset missing frame count
            
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw the footprint trail (line connecting the past centroids)
            for j in range(1, len(footprints[track_id])):
                if footprints[track_id][j - 1] is None or footprints[track_id][j] is None:
                    continue
                # Draw a line between the previous and current points
                cv2.line(img, footprints[track_id][j - 1], footprints[track_id][j], (0, 255, 0), 2)
    
    # Update missing frame counts for all tracks
    for track_id in list(footprints.keys()):
        if track_id not in current_track_ids:
            missing_frames[track_id] += 1
            if missing_frames[track_id] > max_missing_frames:
                # Remove the track if the person is missing for too many frames
                footprints.pop(track_id)
                missing_frames.pop(track_id)
    
    # Show the image with bounding boxes and footprints
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
