import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os

path = 'D:\Attendance System with Face Recognition\Images'
mylist = os.listdir(path)
mylist  # create list with all database images

images = []
classname = []

for cls in mylist:
    currentimg = cv2.imread(f'{path}/{cls}')
    images.append(currentimg)  # list of images
    # slpit "elon.jpg" to "elon" and ".jpg"
    classname.append(os.path.splitext(cls)[0])

# print(classname)


def findEncoding(images):
    encodedimg = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeing = face_recognition.face_encodings(img)[0]
        encodedimg.append(encodeing)
    return encodedimg


knownEncodelist = findEncoding(images)

print('Enconding completed!')


def markattendance(name):
    with open('Attendance.csv', 'r+') as f:
        MyDataList = f.readlines()
        namelist = []

        for line in MyDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')


    # Get images from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resize captured img
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # convert

    # in case of multiple faces in frame.
    facesInCurrrentFrame = face_recognition.face_locations(imgS)
    # encode indentified face.
    encodeInCurrrentFrame = face_recognition.face_encodings(
        imgS, facesInCurrrentFrame)

    # compare all the faces of current frame with out list of faces.

    for encodeface, facelocation in zip(encodeInCurrrentFrame, facesInCurrrentFrame):
        matches = face_recognition.compare_faces(knownEncodelist, encodeface)
        # output:list->size= knowimages(images in our database). Gives us distance of currentface with all the faces of imagelist(database).
        facedistance = face_recognition.face_distance(
            knownEncodelist, encodeface)
        # lower the value of distance closer the face.
        # print(facedistance)
        # give us the index of image with min distance.
        matchindex = np.argmin(facedistance)

        if matches[matchindex]:
            name = classname[matchindex].upper()
            print(name)
            y1, x2, y2, x1 = facelocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markattendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
