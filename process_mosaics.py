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
for dirpath, dirnames, filenames in os.walk("./maps/accra"):
  for filename in [f for f in filenames if f.endswith(".png")]:
    filelist.append(os.path.join(dirpath,filename))
    files.append(filename)
print(files)
print('\n')
print(filelist)

for idx, f in enumerate(filelist):
    with open(f, 'r'):
        image = Image.open(f)
        data = np.asarray(image)
        print("SHAPE: ", data.shape)

        # turn into multichannel
        if data.ndim == 2:
            data = np.stack((data,)*3, axis=-1)
        elif data.shape[2] == 4:
            data = data[:,:,0:3] # get rid of alpha layer
        print("SHAPE: ", data.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB

        # THIS gets axis in he proper order, does not create tiled 9x9 image !!!
        #data = np.moveaxis(data, 0, -1)  
        #print("IMAGES SHAPE AXIS SWAPPED: ", data.shape)
        # re-shape to get 1D array for each layer (a*b, number of bands)
        img = data.reshape((-1, 3))
        print("IMAGE RE-SHAPED: ", img.shape)
        img = np.float32(img)  # convert to 32 bit float for K-Means

        # Uncomment this for testing the mosaic geotif
        # Canny edge settings also seem to work better with 20-25 as arguments
        # Don't forget to comment out the overlay edge to original image code because it isn't implemented yet for geotiffs

        # ------------------------- Clustering through K-Means
        attempts = 10
        K = 4
        ret, label, center = cv2.kmeans(
            img, K, None, None, attempts, cv2.KMEANS_PP_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        # print("RES: ", res.shape)
        # print(images.reshape((images.shape[1], images.shape[2], 3)).shape)
        clusterImg = res.reshape(data.shape)
        im = Image.fromarray(clusterImg)
        filename = os.path.splitext(files[idx])[0]
        im.save('./maps/'+filename+"_quantized.png")

        # ------------------------- Blur Images
        blurredImg = cv2.GaussianBlur(clusterImg, (15, 15), 0)
        im = Image.fromarray(blurredImg)
        im.save('./maps/'+filename+"_blurred.png")

        # ------------------------- Edge detecting with Canny
        # using float here since np.nan requires floating points
        edge = cv2.Canny(blurredImg, 2, 5).astype("float")
        print("EDGE SHAPE: ", edge.shape)
        print("EDGE: ", edge)
        # Not entirely sure why he wanted to replace 0s with "Not a Number" values
        # edge[edge == 0] = np.nan #np.nan only exists with floating-point data types
        edge = np.uint8(edge)  # converting back to 8 bit for viewing (0-255)
        # print("EDGE np.nan: ", edge)
        # print("List all indices that have nan value: ", np.argwhere(np.isnan(edge)))
        image = Image.open(f)
        bckgndImg = np.asarray(image, dtype=np.uint8)
        if bckgndImg.ndim == 2:
            bckgndImg = np.stack((bckgndImg,)*3, axis=-1)
        elif bckgndImg.shape[2] == 4:
            bckgndImg = bckgndImg[:,:,0:3] # get rid of alpha layer

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
        im = Image.fromarray(overlayedImg)
        im.save('./maps/'+filename+"_edge_overlay.png")


