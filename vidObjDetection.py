import cv2
import sys
import winsound

#giving path of HaarCascade classifier files
cascPath1="C:/Users/haarcascade_fullbody.xml"
cascPath3="C:/Users/haarcascade_frontalface_default.xml"
cascPath4="C:/Users/haarcascade_car.xml"

#loading all classifiers
bodyCascade = cv2.CascadeClassifier(cascPath1)
faceCascade=cv2.CascadeClassifier(cascPath3)
carCascade=cv2.CascadeClassifier(cascPath4)

#start capturing from webcam
video_capture = cv2.VideoCapture(0)

#start capturing from video file
#video_capture = cv2.VideoCapture("path of video file")

freq=2500
dur=500

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect bjects of different sizes and return a list of rectangles

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )    

    frontal = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    cars = carCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    #for objects detected, raise beep alarm
    if (faces!=() or cars!=() or frontal!=()):
   
        winsound.Beep(freq,dur)

    #Draw a rectangle around the detected objects
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (fx, fy, fw, fh) in frontal:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

    for (cx, cy, cw, ch) in cars:
        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    #press 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

#to run the program, type the following in cmd:
# python vid1.py
