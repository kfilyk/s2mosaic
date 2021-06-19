# from glob import glob
import glob
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import rasterio
import cv2  # pip install opencv-contrib-python
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pickle

N_OPTICS_BANDS = 4


# note: dont use aerosol optical thickness (AOT) nor water vapour (WVP) bands


tileFolderPath = "./maps/"
tilePaths = []
print("Using Tiles:")
for tilePath in os.listdir(tileFolderPath):  # Get each directory's file path
    if '.SAFE' in tilePath:
        tilePaths.append(tileFolderPath+tilePath)
        print(tilePath)
print("\n")


jp2Paths = []  # Will be made into a 2D array
for i in range(len(tilePaths)):
    jp2Paths.append([])
    # for j in range(3):
    #     jp2Paths[i].append(0)

# For each tile find the jp2 files in
i = 0
for tilePath in tilePaths:
    print("Tile " + tilePath + " contains jp2s:")
    wildcard = glob.glob(tilePath+"/GRANULE/L*/IMG_DATA/R60m/")
    for BandPath in wildcard:  # Should return a single file path to the 10m band directory
        # print(tenMeterBandDirectoryPath)
        for file in os.listdir(BandPath):
            if '_B02' in file or '_B03' in file or '_B04' in file or '_B08' in file:  # RGB + IR
                jp2Paths[i].append(BandPath+file)
                print(BandPath+file)

    print("\n")
    i += 1

images = []
im_vecs = []
for tile in jp2Paths:
    for file in tile:
        # im = Image.open(file)
        # im.show()
        images.append(rasterio.open(file).read(1))
        # data.append(jp2.read(1)) # rasterio
    break  # just get first!! only using on tile currently.

# img_zero = np.array(images[0])
# fig, ax = plt.subplots(1, 1, figsize=(20, 20))
images = np.array(images)
print("IMAGES SHAPE: ", images.shape)
# THIS gets axis in he proper order, does not create tiled 9x9 image !!!
images = np.moveaxis(images, 0, -1)

# plt.imshow(images)
# plt.show()

vectorized = images.reshape((-1, 3))
vectorized = np.float32(vectorized)

print(vectorized.shape)
# vectorized = np.float32(vectorized)

attempts = 2
ret, label, center = cv2.kmeans(
    vectorized, 3, None, None, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
# print("RES: ", res.shape)
# print(images.reshape((images.shape[1], images.shape[2], 3)).shape)
result_image1 = res.reshape(images.shape)
print("RES IMG: ", result_image1.shape)

plt.figure(figsize=(1, 1))
plt.imshow(result_image1)

plt.tight_layout()
plt.savefig('clusters.png')
plt.show()


# this implementation using sklearn probably still works; just switched to cv2 for testing
"""
models = []
model = KMeans(n_clusters=3, max_iter=1, verbose=1)
model.fit(vectorized)
models.append(model)
print("Done")
cluster = model.predict(vectorized)
print(cluster)  # prints ~100,000,000 values for each pixel in 10980x10980 array

with open("kmeans_model.pkl", "wb") as f:  # saves the model so we dont have to rerun necessarily
    pickle.dump(model, f)


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
print(cluster.reshape(images[0].shape).shape)
ax.imshow(cluster.reshape(images[0].shape), cmap="tab20_r")

plt.tight_layout()
plt.savefig('clusters.png')
plt.show()
"""


"""
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
"""
