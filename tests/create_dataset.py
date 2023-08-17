import os
import sys
import cv2
import json
import imutils

class CreateDataset:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

        with open("tests\path.json", encoding='utf-8') as path_file:
                    path = json.load(path_file)

        self.frontal_face_cascade = cv2.CascadeClassifier(path["FrontalCascade"])

        self.face_id = input("Please, enter user id ==> ")
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        self.picture_counter = 0

    def transform_image(self):
        ret, frame = self.capture.read()
        frame = imutils.resize(frame, 640)
        self.grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        faces = self.frontal_face_cascade.detectMultiScale(self.grayscale_frame, 1.2, 10)

        for (x, y, w, h) in faces:
            cv2.rectangle(self.grayscale_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.picture_counter += 1

            def save_faces():
                cv2.imwrite("./module/dataset/User." + str(self.face_id) + '.' +  
                str(self.picture_counter) + ".jpg", self.grayscale_frame[y:y+h,x:x+w])

            save_faces()

            if self.picture_counter >= 30:
                 print("\n [INFO] Exiting Program!")
                 sys.exit()
    
    def display_frame(self):
        cv2.imshow("Frame", self.grayscale_frame)

        k = cv2.waitKey(100) & 0xff 
        if k == 27:
            sys.exit()

    def run(self):
        self.transform_image()
        self.detect_faces()
        self.display_frame()

create_data = CreateDataset()

if __name__ == "__main__":
    while True:
        create_data.run()



