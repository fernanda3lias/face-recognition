import cv2
import sys
import json
import numpy as np

class FaceDetector:
    def __init__(self):
        with open("tests\path.json", encoding='utf-8') as path_file:
            path = json.load(path_file)  
  
        self.frontal_face_cascade = cv2.CascadeClassifier(path["FrontalCascade"])
        self.capture = cv2.VideoCapture(path["CameraID"])

    def transform_image(self) -> np.ndarray:
        ret, frame = self.capture.read()
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

    def display_image(self, faces:np.array, frame:np.ndarray):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi = frame[y:y+h, x:x+w]

        cv2.imshow("Face Recognition", frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            sys.exit()

    def run(self):
        g = self.transform_image()
        f = self.detect_faces(frame=g)
        d = self.display_image(faces=f, frame=g)

if __name__ == "__main__":
    face_detection = FaceDetector()

    while True:
        face_detection.run()

