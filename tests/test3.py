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

# Define colors for different age groups and genders
color_mapping = {
    'Male': {
        '(0-2)': (0, 0, 255),      # Red
        '(4-6)': (0, 128, 255),    # Orange
        '(8-12)': (0, 255, 255),   # Yellow
        '(15-20)': (0, 255, 0),    # Green
        '(25-32)': (255, 0, 0),    # Blue
        '(38-43)': (255, 0, 255),  # Magenta
        '(48-53)': (128, 0, 128),  # Purple
        '(60-100)': (255, 255, 255) # White
    },
    'Female': {
        '(0-2)': (255, 0, 0),      # Light Blue
        '(4-6)': (255, 165, 0),    # Light Orange
        '(8-12)': (255, 255, 0),   # Light Yellow
        '(15-20)': (0, 255, 255),   # Light Green
        '(25-32)': (0, 0, 255),     # Light Red
        '(38-43)': (255, 192, 203), # Pink
        '(48-53)': (186, 85, 211),  # Orchid
        '(60-100)': (0, 128, 128)   # Teal
    }
}

# Open the video file
cap = cv2.VideoCapture("supermarket.mp4")

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a blank canvas for the 2D tracks
canvas_height, canvas_width = 800, 800  # You can adjust this
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_fps = 30  # Frames per second
out = cv2.VideoWriter('canvas_output_with_age_gender.mp4', fourcc, output_fps, (canvas_width, canvas_height))

# Load the YOLOv8 model
model = YOLO('yolo/yolov8x.pt')

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30)

# Dictionary to store the history of positions (footprints) for each track_id
footprints = {}

# Dictionary to store a unique color for each track_id
colors = {}

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

# Function to detect age and gender
def detect_age_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))
    
    # Gender detection
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    
    # Age detection
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    
    return gender, age

# Main loop to process the video
while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    # Run YOLOv8 on the image
    results = model(img, stream=True)
    
    # Initialize a list to hold detections for tracking
# Initialize a list to hold detections for tracking
detections = []

for result in results:
    boxes = result.boxes  # Access to xyxy, conf, cls

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = box.cls[0].item()  # Class label
        
        if cls == 0:  # Assuming 0 is the class ID for 'person'
            # Clamp the bounding box to be within the image dimensions
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame_width, int(x2))
            y2 = min(frame_height, int(y2))

            # Only proceed if the face is valid
            if face.size > 0:  # Check if the face image is valid
                gender, age = detect_age_gender(face)
                
                # Append the detection in the correct format
                detections.append([x1, y1, x2, y2, conf])

# Update tracker with detections
        tracked_objects = tracker.update_tracks(detections, frame=img)
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class label
            
            if cls == 0:  # Assuming 0 is the class ID for 'person'
                # Clamp the bounding box to be within the image dimensions
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(frame_width, int(x2))
                y2 = min(frame_height, int(y2))
                
                face = img[y1:y2, x1:x2]

                # Only proceed if the face is valid
                if face.size > 0:  # Check if the face image is valid
                    gender, age = detect_age_gender(face)

                # Prepare detection for tracker
                detections.append((x1, y1, x2, y2, conf))

    # Update tracker with detections
    tracked_objects = tracker.update_tracks(detections, frame=img)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = map(int, track.to_ltwh())
        track_id = int(track.track_id)

        # Calculate the centroid of the bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Extract the face region from the image for age and gender detection
        face = img[y1:y2, x1:x2]
        gender, age = detect_age_gender(face)

        # Determine color based on gender and age
        color = color_mapping[gender][age]

        # Draw a colored dot on the canvas
        canvas[cy % canvas_height, cx % canvas_width] = color

    cv2.imshow("2D Tracks", canvas)
    out.write(canvas)
    cv2.imshow("Original Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
