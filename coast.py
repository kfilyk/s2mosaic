# from glob import glob
import glob
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2  # pip install opencv-contrib-python
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib as mpl
import pickle

N_OPTICS_BANDS = 4


files = []
filelist = []
for dirpath, dirnames, filenames in os.walk("./maps"):
  for filename in [f for f in filenames if f.endswith(".png")]:
    filelist.append(os.path.join(dirpath,filename))
    files.append(filename)


jp2Paths = []  # Will be made into a 2D array
for i in range(len(tilePaths)):
    jp2Paths.append([])
    # for j in range(3):
    #     jp2Paths[i].append(0)

# For each tile find the jp2 files in
i = 0
for tilePath in tilePaths:
    """
    print("Tile " + tilePath + " contains jp2s:")
    wildcard = glob.glob(tilePath+"/GRANULE/L*/IMG_DATA/R60m/")
    # wildcard = glob.glob("/GRANULE/L2A_T30NUL_A022156_20210603T103702/IMG_DATA/R60m/")
    for BandPath in wildcard:  # Should return a single file path to the 10m band directory
        # print(tenMeterBandDirectoryPath)
        for file in os.listdir(BandPath):
            if '_B02' in file or '_B03' in file or '_B04' in file or '_B08' in file:  # RGB + IR
                jp2Paths[i].append(BandPath+file)
                print(BandPath+file)

    print("\n")
    i += 1
    """
    print("Tile " + tilePath + " contains jpgs:")
    wildcard = glob.glob(tilePath+"/GRANULE/L*/IMG_DATA/R60m/")
    # wildcard = glob.glob("/GRANULE/L2A_T30NUL_A022156_20210603T103702/IMG_DATA/R60m/")
    for BandPath in wildcard:  # Should return a single file path to the 10m band directory
        # print(tenMeterBandDirectoryPath)
        for file in os.listdir(BandPath):
            if '_B02' in file or '_B03' in file or '_B04' in file or '_B08' in file:  # RGB + IR
                jp2Paths[i].append(BandPath+file)
                print(BandPath+file)

    print("\n")
    i += 1

images = []
for tile in jp2Paths:
    for file in tile:
        # im = Image.open(file)
        # im.show()
        images.append(rasterio.open(file).read(1))
        # data.append(jp2.read(1)) # rasterio
    break  # just get first!! only using on tile currently.


# Uncomment this for testing the mosaic geotif
# Canny edge settings also seem to work better with 20-25 as arguments
# Don't forget to comment out the overlay edge to original image code because it isn't implemented yet for geotiffs
# images = rasterio.open("./retrieved_imagery.tif").read()

images = np.array(images)
print("IMAGES SHAPE: ", images.shape)
# THIS gets axis in he proper order, does not create tiled 9x9 image !!!
images = np.moveaxis(images, 0, -1)  # (3, a, b) -> (b, a, 3)
print("IMAGES SHAPE AXIS SWAPPED: ", images.shape)

# plt.imshow(images)
# plt.show()

# re-shape to get 1D array for each layer (a*b, number of bands)
vectorized = images.reshape((-1, 3))
print("IMAGES RE-SHAPED: ", vectorized.shape)

vectorized = np.float32(vectorized)  # convert to 32 bit float for K-Means

# ------------------------- Clustering through K-Means
attempts = 10
K = 4
ret, label, center = cv2.kmeans(
    vectorized, K, None, None, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
# print("RES: ", res.shape)
# print(images.reshape((images.shape[1], images.shape[2], 3)).shape)
clusterImg = res.reshape(images.shape)
print("RESULT IMAGE: ", clusterImg.shape)

# Show K-Means image
clusteredImgFig, clusteredImgAx = plt.subplots(1, 1, figsize=(10, 10))
clusteredImgAx.imshow(clusterImg,  cmap="tab20_r")
clusteredImgAx.set_title("K-Means Image")
plt.tight_layout()
plt.savefig('clusters.png')

# ------------------------- Blur Images
blurredImg = cv2.GaussianBlur(clusterImg, (15, 15), 0)
print("BLURRED IMG SHAPE:", blurredImg.shape)

# Show blur image
blurredImgFig, blurredImgAx = plt.subplots(1, 1, figsize=(10, 10))
blurredImgAx.set_title("Blurred Image")
blurredImgAx.imshow(blurredImg)
plt.savefig('blurred.png')

# ------------------------- Edge detecting with Canny
# using float here since np.nan requires floating points
edge = cv2.Canny(blurredImg, 2, 5).astype("float")
print("EDGE SHAPE: ", edge.shape)
print("EDGE: ", edge)
# Not entirely sure why he wanted to replace 0s with "Not a Number" values
# edge[edge == 0] = np.nan #np.nan only exists with floating-point data types
edge = np.uint8(edge)  # converting back to 8 bit for viewing (0-255)
# print("EDGE np.nan: ", edge)
# print("List all indicies that have nan value: ", np.argwhere(np.isnan(edge)))


# For when we plot edges on rgb image
bckgndImgPath = "./maps/S2B_MSIL2A_20210603T102559_N0300_R108_T30NTL_20210603T124634.SAFE/GRANULE/L2A_T30NTL_A022156_20210603T104943/IMG_DATA/R60m/T30NTL_20210603T102559_TCI_60m.jp2"
bckgndImg = mpimg.imread(bckgndImgPath)
print("OVERLAYBCKGND IMG SHAPE: ", bckgndImg.shape)


# Make edge into 3 channel, bgr or rgb image
# RGB for matplotlib, BGR for imshow()
rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
# Dilate the lines so we can see them clearer on the overlay
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thicBoiEdgeRGB = cv2.dilate(rgb, kernel, iterations=1)
# Now all edges are white (255,255,255). to make it red, multiply with another array:
thicBoiEdgeRGB *= np.array((1, 0, 0), np.uint8)  # Leave R = 1, G, B = 0
# Overlay
overlayedImg = np.bitwise_or(bckgndImg, thicBoiEdgeRGB)


# Show edge images
rawEdgeImg, (rawEdgeImgax) = plt.subplots(1, 1, figsize=(14, 14))
rawEdgeImgax.set_title("Raw Edge Img")
rawEdgeImgax.imshow(edge, cmap="binary")
overlayedImgax.set_title("Overlayed Edge Img")
overlayedImgax.imshow(overlayedImg, cmap='Set3_r')
plt.savefig('rawedge_and_overlay')
# rawEdgeImgax.imshow(rawEdge, cmap=mpl.cm.jet_r, interpolation='nearest')


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
