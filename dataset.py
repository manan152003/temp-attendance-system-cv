import cv2
import numpy as np

 # using haar cascade classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# taking  inputs
name = input("Enter your Name: ")
roll = input(f'Enter your Roll Number {name}: ')

# storing roll in list
with open('roll.csv', 'a') as f:
    # to write name and roll
    f.writelines(f'{name}, {roll}\n')


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        only_face = img[y:y+h, x:x+w]

    return only_face

cap = cv2.VideoCapture(0)
count=0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count=count+1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_path = 'dataset/'+ str(name) +str(count)+'.jpg'
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Register Cam', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)== 27 or count==100:
        break

cap.release()
cv2.destroyAllWindows()

print(f'Dataset Collection of {name} completed')
