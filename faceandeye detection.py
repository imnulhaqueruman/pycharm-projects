import cv2
import numpy as np
# we point opencv's cascadeclassifier function to where our
# classifier (xml file format ) is stored

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

# load our image
img = cv2.imread('emon.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_classifiermport cv2
import numpy as np
# we point opencv's cascadeclassifier function to where our
# classifier (xml file format ) is stored

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

# load our image
img = cv2.imread('emon.jpg')
gray = cv2.c.detectMultiScale(gray,1.3,5)

if faces is ():
    print('No faces found')

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x, y) , (x+w, y+h), (120,0,255), 2)
    #cv2.imshow('Face detect', img)
    #cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h , x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray,1.3,5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.imshow('eye_detection', img)
cv2.imshow('images',img)
cv2.waitKey(0)
cv2.destroyAllWindows()