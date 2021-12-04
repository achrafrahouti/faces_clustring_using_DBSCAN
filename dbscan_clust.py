from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle
import cv2

print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

print("[INFO] clustering...")
clt = DBSCAN(eps=0.5,min_samples=2, metric="euclidean", n_jobs=None)
clt.fit(encodings)
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
for labelID in labelIDs:
        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        print("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(clt.labels_ == labelID)[0]
        faces = []


        # loop over the sampled indexes
        for i in idxs:   
                # load the input image and extract the face ROI
                image = cv2.imread(data[i]['imagePath'])
                (top, right, bottom, left) = data[i]["loc"]
                face = image[top:bottom, left:right]
                # force resize the face ROI to 96x96 and then add it to the
                # faces montage list
                face = cv2.resize(face, (96, 96))
                faces.append(face)
        # create a montage using 96x96 "tiles" with 5 rows and 5 columns
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        cv2.imwrite(str(labelID)+".jpg", montage)
        # show the output montage
        title = "Face ID #{}".format(labelID)
        title = "Unknown Faces" if labelID == -1 else title
        cv2.imshow(title, montage)
        cv2.waitKey(0)