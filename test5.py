import cv2
import numpy as np
from scipy.spatial import Delaunay
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from collections import Counter

# Load pre-trained age and gender models
gender_net = cv2.dnn.readNet('Gender-and-Age-Detection/gender_net.caffemodel', 'Gender-and-Age-Detection/gender_deploy.prototxt')
age_net = cv2.dnn.readNet('Gender-and-Age-Detection/age_net.caffemodel', 'Gender-and-Age-Detection/age_deploy.prototxt')
gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Age and gender model parameters
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Open the video file
cap = cv2.VideoCapture("supermarket.mp4")

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a blank canvas for the 2D tracks
canvas_height, canvas_width = 720, 2000
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Load the YOLOv8 model
model = YOLO('yolo/yolov8x.pt')
model.to("cuda")

# Initialize video writer for the canvas
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
canvas_writer = cv2.VideoWriter('test.mp4', fourcc, 30, (canvas_width, canvas_height))



# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30)

# Dictionaries to store the history of positions (footprints), gender, and age for each track_id
footprints = {}
gender_info = {}
age_info = {}
colors = {}

# Initialize counters for male and female
gender_counts = {'Male': 0, 'Female': 0}
total_detected_count = 0

# Set to track all detected IDs for unique count
detected_ids = set()

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

# Function to detect age and gender
def detect_age_gender(face_img):
    # Detect gender
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (104, 117, 123), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    
    # Detect age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    return gender, age

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

# Initialize counts for unique entries
while True:
    unique_male_count = 0
    unique_female_count = 0
    
    success, img = cap.read()
    if not success:
        break  # End of video

    # Run YOLOv8 on the image
    results = model(img, stream=True)

    # Initialize a list to hold detections for tracking
    detections = []

    # Count the number of detections in this frame
    total_detected_count = 0

    for result in results:
        boxes = result.boxes  # Access to xyxy, conf, cls
        
        # Convert boxes to a format compatible with DeepSORT tracker
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            
            # Filter by person class (class 0 is 'person')
            if cls == 0:
                total_detected_count += 1  # Increment total detected count for every person detected
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

        # Calculate the centroid of the bounding box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Extract face for age and gender detection
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            continue  # Skip empty face crops

        # Run gender and age detection every 10 frames
        if frame_count % 10 == 0:
            gender, age = detect_age_gender(face_img)
            # Store gender and age for future reference
            gender_info[track_id] = gender
            age_info[track_id] = age
        else:
            # Use the previously stored gender and age if not running detection
            gender = gender_info.get(track_id, None)
            age = age_info.get(track_id, None)

        # Get color based on gender
        if gender is not None:
            color = get_color(track_id, gender)
        else:
            color = (0, 0, 0)

        # Track total unique IDs detected
        detected_ids.add(track_id)

        # Count unique males and females based on gender
        if gender == 'Male':
            unique_male_count += 1
        elif gender == 'Female':
            unique_female_count += 1

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
            cv2.circle(canvas, (transformed_cx, transformed_cy), 4, color, -1)

        # Draw the bounding box and display the gender and age
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if gender is not None and age is not None:
            cv2.putText(img, f"{gender}, {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Clear the previous counts on the canvas
    cv2.rectangle(canvas, (900, 0), (1550, 150), (0, 0, 0), -1)  # Clear area for counts
    cv2.rectangle(canvas, (900, 60), (1550, 250), (0, 0, 0), -1)  # Clear area for counts


    # Draw the total unique count on the canvas
    # Display the total number of detected people (from YOLO) on the canvas
    cv2.putText(canvas, f"Total Detected People: {total_detected_count}", (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(canvas, f"Unique Males: {unique_male_count}", (900, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(canvas, f"Unique Females: {unique_female_count}", (900, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 255), 1)
     

    
        # Draw rectangles
    cv2.rectangle(canvas, (710, 320) , (720, 550), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (300, 230), (400, 450), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (810, 150), (820, 600), (255, 255, 255), -1, cv2.LINE_AA)

    
    # Calculate the most prevalent age from tracked IDs
    if age_info:
        age_list = [age_info[track_id] for track_id in age_info if track_id in footprints]
        if age_list:
            most_common_age = Counter(age_list).most_common(1)
            if most_common_age:
                prevalent_age = most_common_age[0][0]
                cv2.putText(canvas, f"Most Prevalent Age: {prevalent_age}", (900, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the images
    cv2.imshow("Tracked People", img)
    cv2.imshow("Canvas", canvas)
    canvas_writer.write(canvas)
    frame_count += 1  # Increment frame count
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
canvas_writer.release()
cap.release()
cv2.destroyAllWindows()
