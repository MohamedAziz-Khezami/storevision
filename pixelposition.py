# importing the module 
import cv2 

# function to display the coordinates of 
# the points clicked on the image 
def click_event(event, x, y, flags, params): 

    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        # displaying the coordinates on the Shell 
        print(x, ' ', y) 

        # displaying the coordinates on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2) 
        cv2.imshow('First Frame', img) 

    # checking for right mouse clicks	 
    if event == cv2.EVENT_RBUTTONDOWN: 
        # displaying the coordinates on the Shell 
        print(x, ' ', y) 

        # displaying the coordinates on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2) 
        cv2.imshow('First Frame', img) 

# driver function 
if __name__ == "__main__": 
    # Open the video file 
    video = cv2.VideoCapture("supermarket.mp4")  # Replace with your video path
    
    # Read the first frame from the video
    success, img = video.read() 

    if not success:
        print("Failed to read the video file")
        exit()

    # displaying the first frame 
    cv2.imshow('First Frame', img) 

    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('First Frame', click_event) 

    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 

    # release the video capture and close the window 
    video.release() 
    cv2.destroyAllWindows()
