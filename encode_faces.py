# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:08:35 2021

@author: Achraf Rahouti
"""
from imutils import paths
import face_recognition
import pickle
import cv2
import os
ENCODINGS_PATH=os.path.join(os.getcwd(),'encodings.pickle')
print(ENCODINGS_PATH)

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("data"))
data = []

for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	print(imagePath)

	image = cv2.imread(imagePath)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(image,model='cnn')

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(image, boxes)

	d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
		for (box, enc) in zip(boxes, encodings)]
	data.extend(d)

print("[INFO] serializing encodings...")
f = open('encodings.pickle', "wb")
f.write(pickle.dumps(data))
f.close()
print("Encodings of images saved in {}".format(ENCODINGS_PATH))