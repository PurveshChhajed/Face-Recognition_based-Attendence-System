# Import OpenCV2 for image processing
import cv2
import os
import time
#import cv2.cv as cv

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
face_id = input("enter your face id")
# Start capturing video 
#vid_cam = cv2.VideoCapture(1)#,cv2.CAP_DSHOW)
vid_cam = cv2.VideoCapture("http://192.168.43.49:8080/video")
vid_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 8000)   #7680, 4320
vid_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize sample face image
count = 0

assure_path_exists("data_new/")

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:
        time.sleep(0.4)
        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("data_new/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    
    elif count>=10:
        print("Successfully Captured")
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
