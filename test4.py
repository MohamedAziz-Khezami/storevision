import cv2
import numpy as np
import random
from scipy.spatial import Delaunay
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Load the pre-trained models for age and gender detection
age_net = cv2.dnn.readNetFromCaffe('Gender-and-Age-Detection/age_deploy.prototxt', 'Gender-and-Age-Detection/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('Gender-and-Age-Detection/gender_deploy.prototxt', 'Gender-and-Age-Detection/gender_net.caffemodel')

# Age and gender labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']




# Open the video file
cap = cv2.VideoCapture("supermarket.mp4")

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a blank canvas for the 2D tracks
canvas_height, canvas_width = 800, 800  # You can adjust this
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Load the YOLOv8 model
model = YOLO('yolo/yolov8x.pt')

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30)

# Dictionary to store the history of positions (footprints) for each track_id
footprints = {}

# Dictionary to store a unique color for each track_id
colors = {}

# Legend positions
legend_x = canvas_width - 120
legend_y = 20
legend_spacing = 30


# Define the rectangle position and size
rect_start = (100, 100)  # Top-left corner (x, y)
rect_end = (300, 200)    # Bottom-right corner (x, y)
rect_color = (255, 255, 255)  # White color in BGR
rect_thickness = -1  # Fill the rectangle

# Define the 3D points (corners of the irregular shape)
points_3d = np.float32([
    [20, 708],  # bottom-left corner
    [1266, 167],  # bottom-right corner
    [8, 14],  # top-left corner
    [436, 9],  # top-right corner
    [1103, 709],  # another point
])

# Corresponding points on the 2D canvas
points_2d = np.float32([
    [0, canvas_height],    # bottom-left corner in 2D space
    [canvas_width, canvas_height],  # bottom-right corner in 2D space
    [0, 0],      # top-left corner in 2D space
    [canvas_width, 0],     # top-right corner in 2D space
    [400, 800],  # corresponding point on canvas
])

# Delaunay triangulation for the 3D points
triangles = Delaunay(points_3d[:, :2])

# Function to get the affine transformation for each triangle
def get_affine_transform(tri_points_3d, tri_points_2d):
    M = cv2.getAffineTransform(tri_points_3d, tri_points_2d)
    return M

# Function to apply affine transformation to a point inside a triangle
def apply_affine_transform(point, M):
    point = np.array([point[0], point[1], 1]).reshape(3, 1)
    transformed_point = np.dot(M, point)
    return int(transformed_point[0]), int(transformed_point[1])

# Function to get color for track_id
def get_color(track_id):
    if track_id not in colors:
        colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return colors[track_id]

# Threshold for the minimum movement distance to draw the next dot
MIN_MOVE_DISTANCE = 40

while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    # Run YOLOv8 on the image
    results = model(img, stream=True)
    
    # Initialize a list to hold detections for tracking
    detections = []

    # Iterate through each result in the stream
    for result in results:
        boxes = result.boxes  # Access to xyxy, conf, cls
        
        # Convert boxes to a format compatible with DeepSORT tracker
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class label
            
            # Filter by person class (class 0 is 'person')
            if cls == 0:
                detections.append(([x1, y1, x2, y2], conf))

    # Update tracker with detections
    tracked_objects = tracker.update_tracks(detections, frame=img)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = map(int, track.to_ltwh())
        track_id = int(track.track_id)

        color = get_color(track_id)
        
        # Calculate the centroid of the bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Extract the face region from the image for age and gender detection
        face = img[y1:y2, x1:x2]
        if face.size > 0:  # Ensure the face is valid
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))

            # Gender detection
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            # Age detection
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
        else:
            continue  # Skip if face detection failed

        # Add the current centroid to the footprint history
        if track_id not in footprints:
            footprints[track_id] = []
        else:
            # Only draw a dot if the centroid moved significantly
            prev_cx, prev_cy = footprints[track_id][-1]
            distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            if distance < MIN_MOVE_DISTANCE:
                continue  # Skip drawing if the movement is too small

        footprints[track_id].append((cx, cy))

        # Find which triangle the point belongs to
        simplex = triangles.find_simplex([cx, cy])

        if simplex >= 0:  # If the point is inside a triangle
            triangle_3d = points_3d[triangles.simplices[simplex]]  # Get 3D points of the triangle
            triangle_2d = points_2d[triangles.simplices[simplex]]  # Get 2D points of the triangle

            # Get the affine transformation for this triangle
            M = get_affine_transform(triangle_3d.astype(np.float32), triangle_2d.astype(np.float32))

            # Apply affine transformation to the centroid
            transformed_cx, transformed_cy = apply_affine_transform([cx, cy], M)

            # Color code based on gender
            point_color = (255, 0, 0) if gender == 'Female' else (0, 0, 255)  # Red for Female, Blue for male
            cv2.circle(canvas, (transformed_cx, transformed_cy), 4, point_color, -1)
            #cv2.putText(canvas, f"{track_id}: {age}", (transformed_cx + 5, transformed_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1)

    # Draw the legend
    cv2.rectangle(canvas, (legend_x, legend_y), (legend_x + 100, legend_y + 20), (255, 255, 255), -1)
    cv2.putText(canvas, "Legend:", (legend_x + 5, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Male legend
    cv2.rectangle(canvas, (legend_x, legend_y + legend_spacing), (legend_x + 20, legend_y + legend_spacing + 20), (255, 0, 0), -1)
    cv2.putText(canvas, "Male", (legend_x + 30, legend_y + legend_spacing + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Female legend
    cv2.rectangle(canvas, (legend_x, legend_y + 2 * legend_spacing), (legend_x + 20, legend_y + 2 * legend_spacing + 20), (0, 0, 255), -1)
    cv2.putText(canvas, "Female", (legend_x + 30, legend_y + 2 * legend_spacing + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.rectangle(canvas, rect_start, rect_end, rect_color, rect_thickness)



    # Show the 2D canvas with the tracked dots and labels
    cv2.imshow("2D Tracks", canvas)

    # Show the original video frame with bounding boxes
    cv2.imshow("Original Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
