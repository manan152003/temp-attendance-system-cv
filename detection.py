import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time
import csv

name = input("Enter your Name: ")

data_path = 'dataset/'
onlyfiles = [i for i in listdir(data_path) if isfile(join(data_path,i))]

# *************************** getting roll no. ***************************
with open('roll.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    try:
        for line in csv_reader:
            while(True):
                i=0
                i = i+1
                if (line[i] == name):
                    print(f'Roll number : {line[1]} is there in list!')
                    roll = line[1]
    except:
        print(f'Sorry {name}, your name is not present in list')


# *************************** using time module for time ***************************
localtime = time.asctime(time.localtime(time.time()))
# print(localtime)

def markAttendance():
    #  *************************** getting things in excel ***************************
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            f.writelines(f'\n{name}, {roll}, {localtime}')



Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    img_path = data_path + onlyfiles[i]
    images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Dataset Training Completed !!!")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detection(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi

cap = cv2.VideoCapture(0)
while(True):

    ret, frame = cap.read()

    image, face = face_detection(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence) + '% Confident it is ' + name
        cv2.putText(image, display_string, (80, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (123,31,202), 2)

        if confidence > 82:
            cv2.putText(image, f'{roll} is verified! :)', (120,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Attendance Cam', image)
            markAttendance()

        else:
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Attendance Cam', image)

    except:
        cv2.putText(image, "No Face Found", (180, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Attendance Cam', image)
        pass


    if cv2.waitKey(1) == 27 :
        break

cap.release()
cv2.destroyAllWindows()
