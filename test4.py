import cv2
import numpy as np
import random
from scipy.spatial import Delaunay
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Load pre-trained age and gender models

gender_net = cv2.dnn.readNet('Gender-and-Age-Detection/gender_net.caffemodel', 'Gender-and-Age-Detection/gender_deploy.prototxt')
gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Age and gender model parameters
GENDER_LIST = ['Male', 'Female']

# Open the video file
cap = cv2.VideoCapture("supermarket.mp4")

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a blank canvas for the 2D tracks
canvas_height, canvas_width = 720, 1280
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Load the YOLOv8 model
model = YOLO('yolo/yolov8x.pt')
model.to("cuda")
# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30)

# Dictionary to store the history of positions (footprints) for each track_id
footprints = {}

# Dictionary to store a unique color for each track_id and gender information
colors = {}
gender_info = {}

# Define the 3D points (corners of the irregular shape)
points_3d = np.float32([
    [20, 708],  
    [1266, 167],
    [8, 14],    
    [436, 9],   
    [1103, 709]
])

# Corresponding points on the 2D canvas
points_2d = np.float32([
    [0, 800],    
    [800, 800],  
    [0, 0],      
    [800, 0],    
    [400, 800]
])



# Delaunay triangulation for the 3D points
triangles = Delaunay(points_3d[:, :2])

# Function to get the affine transformation for each triangle
def get_affine_transform(tri_points_3d, tri_points_2d):
    return cv2.getAffineTransform(tri_points_3d, tri_points_2d)

# Function to apply affine transformation to a point inside a triangle
def apply_affine_transform(point, M):
    point = np.array([point[0], point[1], 1]).reshape(3, 1)
    transformed_point = np.dot(M, point)
    return int(transformed_point[0]), int(transformed_point[1])

# Function to detect gender and age
def detect_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (104, 117, 123), swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]



    return gender

# Function to get color for track_id based on gender
def get_color(track_id, gender):
    if track_id not in colors:
        if gender == 'Male':
            colors[track_id] = (255, 0, 0)  # Blue for Male
        else:
            colors[track_id] = (255, 0, 255)  # Pink for Female
    return colors[track_id]

# Threshold for the minimum movement distance to draw the next dot
MIN_MOVE_DISTANCE = 20
frame_count = 0  # Initialize the frame counter

while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    # Run YOLOv8 on the image
    results = model(img, stream=True)
    
    # Initialize a list to hold detections for tracking
    detections = []

    for result in results:
        boxes = result.boxes  # Access to xyxy, conf, cls
        
        # Convert boxes to a format compatible with DeepSORT tracker
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            
            # Filter by person class (class 0 is 'person')
            if cls == 0:
                detections.append(([x1, y1, x2, y2], conf))

    # Update tracker with detections
    tracked_objects = tracker.update_tracks(detections, frame=img)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        # Convert from (left, top, width, height) to (x1, y1, x2, y2)
        left, top, width, height = track.to_ltwh()
        x1, y1 = int(left), int(top)
        x2, y2 = int(left + width), int(top + height)
        x1, y1 = max(0, min(frame_width, x1)), max(0, min(frame_height, y1))
        x2, y2 = max(0, min(frame_width, x2)), max(0, min(frame_height, y2))
        track_id = int(track.track_id)

        # Extract face for age and gender detection
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            continue  # Skip empty face crops

        # Run gender detection every 10 frames
        if frame_count % 10 == 0:
            gender = detect_age_gender(face_img)
            # Store gender for future reference
            gender_info[track_id] = gender
        else:
            # Use the previously stored gender if not running detection
            gender = gender_info.get(track_id, None)

        # Get color based on gender
        if gender is not None:
            color = get_color(track_id, gender)
        else :
            color = (0,0,0)

        # Calculate the centroid of the bounding box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Add the current centroid to the footprint history
        if track_id not in footprints:
            footprints[track_id] = []
        else:
            prev_cx, prev_cy = footprints[track_id][-1]
            distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            if distance < MIN_MOVE_DISTANCE:
                continue

        footprints[track_id].append((cx, cy))

        # Find which triangle the point belongs to
        simplex = triangles.find_simplex([cx, cy])

        if simplex >= 0:
            triangle_3d = points_3d[triangles.simplices[simplex]]
            triangle_2d = points_2d[triangles.simplices[simplex]]
            M = get_affine_transform(triangle_3d.astype(np.float32), triangle_2d.astype(np.float32))
            transformed_cx, transformed_cy = apply_affine_transform([cx, cy], M)

            # Draw the point
            cv2.circle(canvas, (transformed_cx, transformed_cy), 4, color , -1)

    # Show the 2D canvas with the tracked dots and labels
    cv2.imshow("2D Tracks", canvas)

    # Show the original video frame with bounding boxes and IDs
    cv2.imshow("Original Video", img)

    # Draw rectangles
    cv2.rectangle(canvas, (710, 320) , (720, 550), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (300, 230), (400, 450), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (810, 150), (820, 600), (255, 255, 255), -1, cv2.LINE_AA)

    # Increment the frame count
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



