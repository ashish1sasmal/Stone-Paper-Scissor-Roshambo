
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import sys
import cv2
import numpy as np
import joblib
import os

class KNN:
    def __init__(self, k_neighbors, n_jobs, imagePaths , imgRes,dataset):
        self.n_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.imagePaths = [x[0] for x in os.walk(imagePaths)][1:]
        self.imgRes = imgRes
        self.dataset = dataset

    def process1(self,img):
        return cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
    
    def load(self, count):
        data = []
        labels =[]
        c=0
        print(self.imagePaths)
        for dir in self.imagePaths:
            images = list(paths.list_images(dir))
            # print(images)
            for (i,imagePath) in  enumerate(images):
                c+=1
                # print(imagePath)
                img = cv2.imread(imagePath)
                label = imagePath.split("/")[1]
                # print(label)
                img = self.process1(img)
                data.append(img)
                labels.append(label)
        print(c)
        if c==count:
            print("[INFO] processed {}/{}".format(c, 4000))
        print(len(data),len(labels))
        return (np.array(data), np.array(labels))

    def classify(self):
        (data, labels) = self.load(4000)
        data = data.reshape((data.shape[0], (self.imgRes**2)*3))

        print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

        le = LabelEncoder()
        labels = le.fit_transform(labels)

        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state = 42)

        print("[INFO] evaluating k-NN classifier...")

        model = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs = self.n_jobs)
        print(len(trainX),len(trainY))
        model.fit(trainX, trainY)

        print(classification_report(testY, model.predict(testX), target_names=le.classes_))

        # model.save("dog-cat-panda.h5")
        joblib.dump(model, f'{self.dataset}.pkl')



        
    


    