# **🛒StoreVision**

StoreVision is an advanced video analytics application designed to track and analyze customer movements in retail environments such as stores, malls, and supermarkets. By providing detailed insights into foot traffic and customer paths, it helps managers optimize store layouts, identify high-traffic "hot spots," and create targeted marketing strategies based on customer behavior.



## **Main Objectives:**

- Identify Hot Spots: Analyze customer movement patterns to pinpoint areas with the highest engagement, useful for optimizing store layouts or positioning advertisements and product displays.
- Customer Flow Analysis: Gain insights into how customers navigate through different areas of the store, identifying choke points or areas where customers spend the most time.
- Demographic Insights: Track the age and gender of customers to create more tailored marketing strategies and optimize product placement.



## Key Features:

- Footprint Tracking: Tracks customer movements using person detection and re-identification across multiple frames.
- Age and Gender Detection: Recognizes the gender and approximate age range of customers to help build demographic insights.
- Heatmap Generation: By monitoring foot traffic over time, StoreVision helps identify the most frequently visited areas in the store.
- Real-Time Display: The tracked movement, along with demographic data, is displayed on a custom canvas.
- GPU-Accelerated Processing: Utilizes CUDA-enabled models to ensure smooth real-time processing, leveraging the power of NVIDIA GPUs for faster inference.


## Technology Stack:

- `OpenCV`: For capturing video and processing frames.
- `DeepSORT`: Real-time multi-object tracking to maintain the identity of customers as they move through the store.
- `YOLO (You Only Look Once)`: Pretrained YOLOv8 model used for detecting customers in the video frames.
- `Delaunay Triangulation`: Used for mapping customer movements onto a 2D canvas to visualize the paths of customers.
- `Age and Gender Models`: DNN-based models for gender and age detection using pre-trained Caffe models.
- `NVIDIA GPU Acceleration`: Used to offload computation to the GPU, speeding up both detection and tracking operations.



## Screenshots

![Capture d’écran 2024-09-25 193353](https://github.com/user-attachments/assets/b72f2689-d8f8-4ab9-a38f-d58f0fff783b)



![Capture d’écran 2024-09-25 193420](https://github.com/user-attachments/assets/4ab556d4-0e9b-48b5-bd6a-3f682f5e792e)



## Feedback

If you have any feedback, please reach out to me at mohamedazizkhezami@gmail.com

