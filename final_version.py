import cv2
import numpy as np
from scipy.spatial import Delaunay
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from collections import Counter


gender_net = cv2.dnn.readNet('Gender-and-Age-Detection/gender_net.caffemodel', 'Gender-and-Age-Detection/gender_deploy.prototxt')
age_net = cv2.dnn.readNet('Gender-and-Age-Detection/age_net.caffemodel', 'Gender-and-Age-Detection/age_deploy.prototxt')
gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


cap = cv2.VideoCapture("supermarket.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_width = 600  
video_height = 400  
video_x = 900  
video_y = 200  



canvas_height, canvas_width = 720, 2000
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)


model = YOLO('yolo/yolov8x.pt')
model.to("cuda")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
canvas_writer = cv2.VideoWriter('test.mp4', fourcc, 30, (canvas_width, canvas_height))



tracker = DeepSort(max_age=30)

footprints = {}
gender_info = {}
age_info = {}
colors = {}

gender_counts = {'Male': 0, 'Female': 0}
total_detected_count = 0

detected_ids = set()

points_3d = np.float32([
    [20, 708],
    [1266, 167],
    [8, 14],
    [436, 9],
    [1103, 709]
])

points_2d = np.float32([
    [0, 800],
    [800, 800],
    [0, 0],
    [800, 0],
    [400, 800]
])

triangles = Delaunay(points_3d[:, :2])

def get_affine_transform(tri_points_3d, tri_points_2d):
    return cv2.getAffineTransform(tri_points_3d, tri_points_2d)

def apply_affine_transform(point, M):
    point = np.array([point[0], point[1], 1]).reshape(3, 1)
    transformed_point = np.dot(M, point)
    return int(transformed_point[0]), int(transformed_point[1])

def detect_age_gender(face_img):
   
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (104, 117, 123), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    
    
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    return gender, age

def get_color(track_id, gender):
    if track_id not in colors:
        if gender == 'Male':
            colors[track_id] = (255, 0, 0)  # Blue for Male
        else:
            colors[track_id] = (255, 0, 255)  # Pink for Female
    return colors[track_id]


MIN_MOVE_DISTANCE = 20
frame_count = 0  

# main loop
while True:
    unique_male_count = 0
    unique_female_count = 0
    
    success, img = cap.read()
    if not success:
        break  

    results = model(img, stream=True)

    detections = []

    total_detected_count = 0

    for result in results:
        boxes = result.boxes  
        

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            
            # Filter by person class (class 0 is 'person')
            if cls == 0:
                total_detected_count += 1  
                detections.append(([x1, y1, x2, y2], conf))

    tracked_objects = tracker.update_tracks(detections, frame=img)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        left, top, width, height = track.to_ltwh()
        x1, y1 = int(left), int(top)
        x2, y2 = int(left + width), int(top + height)
        x1, y1 = max(0, min(frame_width, x1)), max(0, min(frame_height, y1))
        x2, y2 = max(0, min(frame_width, x2)), max(0, min(frame_height, y2))
        track_id = int(track.track_id)

        #centriod
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Extract face 
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            continue  # Skip empty face crops

        if frame_count % 30 == 0:
            gender, age = detect_age_gender(face_img)
            gender_info[track_id] = gender
            age_info[track_id] = age
        else:
            gender = gender_info.get(track_id, None)
            age = age_info.get(track_id, None)


        if gender is not None:
            color = get_color(track_id, gender)
        else:
            color = (255, 255, 255)


        detected_ids.add(track_id)

        if gender == 'Male':
            unique_male_count += 1
        elif gender == 'Female':
            unique_female_count += 1


        if track_id not in footprints:
            footprints[track_id] = []
        else:
            prev_cx, prev_cy = footprints[track_id][-1]
            distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            if distance < MIN_MOVE_DISTANCE:
                continue

        footprints[track_id].append((cx, cy))

        simplex = triangles.find_simplex([cx, cy])

        if simplex >= 0:
            triangle_3d = points_3d[triangles.simplices[simplex]]
            triangle_2d = points_2d[triangles.simplices[simplex]]
            M = get_affine_transform(triangle_3d.astype(np.float32), triangle_2d.astype(np.float32))
            transformed_cx, transformed_cy = apply_affine_transform([cx, cy], M)

            cv2.circle(canvas, (transformed_cx, transformed_cy), 4, color, -1)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if gender is not None and age is not None:
            cv2.putText(img, f"{gender}, {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.rectangle(canvas, (900, 0), (1550, 150), (0, 0, 0), -1)  
    cv2.rectangle(canvas, (900, 60), (1550, 250), (0, 0, 0), -1)  # Clear area for counts



    cv2.putText(canvas, f"Total Detected People: {total_detected_count}", (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(canvas, f"Unique Males: {unique_male_count}", (900, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(canvas, f"Unique Females: {unique_female_count}", (900, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 255), 1)
     

    
    # Draw rectangles
    cv2.rectangle(canvas, (690, 320) , (720, 550), (0,69,255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (300, 230), (400, 450), (128,128,128), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (810, 150), (820, 600), (200, 0, 0), -1, cv2.LINE_AA)
    
    cv2.rectangle(canvas, (830, 10), (835, 1000), (114,128,112), -1 , cv2.LINE_AA) 
    cv2.rectangle(canvas, (10, 10), (830, 15), (114,128,112), -1 , cv2.LINE_AA)   
    
    cv2.rectangle(canvas, (10, 10), (15, 1000), (114,128,112), -1 , cv2.LINE_AA)   

    
    

    
    if age_info:
        age_list = [age_info[track_id] for track_id in age_info if track_id in footprints]
        if age_list:
            most_common_age = Counter(age_list).most_common(1)
            if most_common_age:
                prevalent_age = most_common_age[0][0]
                cv2.putText(canvas, f"Most Prevalent Age: {prevalent_age}", (900, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
    resized_video_frame = cv2.resize(img, (video_width, video_height))
    canvas[video_y:video_y + video_height, video_x:video_x + video_width] = resized_video_frame

    cv2.imshow("Canvas", canvas)
    canvas_writer.write(canvas)
    frame_count += 1  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
canvas_writer.release()
cap.release()
cv2.destroyAllWindows()
