import os
import sys
import cv2
import json
import imutils
import numpy as np

class FaceRecognition:
    def __init__(self):
        with open("tests\path.json", encoding='utf-8') as path_file:
            path = json.load(path_file)  

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(path["TrainedModel"])
        self.frontal_face_cascade = cv2.CascadeClassifier(path["FrontalCascade"])
        self.capture = cv2.VideoCapture(path["CameraID"])

        # Define variables
        self.id = 0
        self.names = ["None", "Fernanda", "Lucas"]

    def transform_image(self) -> np.ndarray:
        ret, frame = self.capture.read()
        frame = imutils.resize(frame, 640)
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return grayscale_frame 
    
    def detect_faces(self, frame) -> np.array:
        faces = self.frontal_face_cascade.detectMultiScale(
            frame,
            scaleFactor = 1.2,
            minNeighbors = 10,
            minSize = (60, 60)
        )

        return faces
    
    def recognize_faces(self, frame, faces):
        id = self.id
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            id, confidence = self.recognizer.predict(frame[y:y+h, x:x+w])

            if (confidence < 100):
                id = self.names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(frame, 
                    str(id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,255), 2)
            cv2.putText(frame,
                    str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,0), 1) 

        return frame
    
    def display_windows(self, frame):
        cv2.imshow('camera',frame) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            sys.exit()

    def run(self):
        g = self.transform_image()
        f = self.detect_faces(frame=g)
        g = self.recognize_faces(frame=g, faces=f)
        self.display_windows(frame=g)

if __name__ == "__main__":
    fr = FaceRecognition()
    while True:
        fr.run()