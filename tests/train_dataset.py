import os
import cv2
import json
import numpy as np
from PIL import Image

class TrainDataset:
    def __init__(self):
        with open("tests\path.json", encoding='utf-8') as path_file:
            path = json.load(path_file) 
        self.dataset_path = path["Dataset"]
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.frontal_face_cascade = cv2.CascadeClassifier(path["FrontalCascade"])
    
    def get_images_and_labels(self):
        image_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')
            numpy_image = np.array(pil_image,'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = self.frontal_face_cascade.detectMultiScale(numpy_image)

            for (x, y, w, h) in faces:
                face_samples.append(numpy_image[y:y+h,x:x+w])
                ids.append(id)

        print(face_samples, ids)

        return face_samples, ids

    def train_dataset(self, faces, ids):
        print ("\n [INFO] Training faces.")
        ids = np.array(ids)
        self.recognizer.train(faces, ids)
        self.recognizer.write("./module/source/trainer.yml")

        print(" [INFO] SUCCESS")

    def run(self):
        fs, ids = self.get_images_and_labels()
        self.train_dataset(faces=fs, ids=ids)

if __name__ == "__main__":
    train_ds = TrainDataset()
    train_ds.run()