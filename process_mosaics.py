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
'''
for dirpath, dirnames, filenames in os.walk("./maps/accra"):
  for filename in [f for f in filenames if f.endswith(".png")]:
    filelist.append(os.path.join(dirpath,filename))
    files.append(filename)
print(files)
print('\n')
print(filelist)
'''
filelist.append('./maps/accra/accra_rgb.png')
files.append('accra_rgb.png')

for idx, f in enumerate(filelist):
    filename = os.path.splitext(files[idx])[0]
    with open(f, 'r'):
        image = Image.open(f)
        img = np.asarray(image)

        # turn into multichannel
        if img.ndim == 2:
            img = np.stack((img,)*3, axis=-1)
        elif img.shape[2] == 4: # get rid of alpha chan
            img = img[:,:,0:3] 

        # ------------------------- Add B09 Band For Cloud Detection 

        b09_image = Image.open('./maps/accra/accra_b09.png')
        b09 = np.asarray(b09_image, dtype = np.float32)
        print("B09 SHAPE: ", b09.shape)
        img= np.insert(img, img.shape[2], b09, axis = 2) # insert b09 band to alpha layer
        print("SHAPE: ", img.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB
        og_shape = img.shape

        # ------------------------- Add CLP Map For Cloud Detection 

        clp_image = Image.open('./maps/accra/accra_clp.png')
        clp = np.asarray(clp_image, dtype = np.float32)
        print("B09 SHAPE: ", clp.shape)
        img= np.insert(img, img.shape[2], b09, axis = 2) # insert b09 band to alpha layer
        print("SHAPE: ", img.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB
        og_shape = img.shape

        # ------------------------- Add B01 Band For Cloud Detection 

        '''
        aerosol = Image.open('./maps/accra/accra_aerosol.png')
        a_data = np.asarray(aerosol, dtype = np.float32)
        print("AERO SHAPE: ", a_data.shape)
        """
        min = np.amin(a_data) 
        a_data = a_data - min
        max = np.amax(a_data)
        a_data = a_data * 255/max
        """
        a_data = a_data.astype(np.uint8)

        im = Image.fromarray(a_data)
        im.save('./maps/'+filename+"_aero.png")        
        a_data = a_data.astype(np.float64)
        '''
        # ------------------------- Grey Intensity 
        '''
        data = data.astype(np.float64)

        grey = ((data[:, :, 0] + data[:, :, 1] + data[:, :, 2] + a_data[:, :])/4)
        min = np.amin(grey)
        grey = grey - min
        max = np.amax(grey)
        grey = grey * 255/max
        grey = np.stack((grey,)*3, axis=-1)
        grey = grey.astype(np.uint8)
        print(grey.shape)
        im = Image.fromarray(grey)
        im.save('./maps/'+filename+"_grey.png")        

        grey_2 = (data[:, :, 0] * data[:, :, 1] * data[:, :, 2] * a_data[:,:])
        grey_2 = np.sqrt(grey_2)
        min = np.amin(grey_2)
        grey_2 = grey_2 - min
        max = np.amax(grey_2)
        grey_2 = grey_2 * 255/max
        grey_2 = np.stack((grey_2,)*3, axis=-1)
        print(grey_2.shape)
        grey_2 = grey_2.astype(np.uint8)
        im = Image.fromarray(grey_2)
        im.save('./maps/'+filename+"_grey_2.png")   

        grey_3 = (data[:, :, 0]*data[:, :, 0] + data[:, :, 1]*data[:, :, 1] + data[:, :, 2]*data[:, :, 2]  + a_data[:, :]*a_data[:, :])
        grey_3 = np.sqrt(grey_3)
        min = np.amin(grey_3)
        grey_3 = grey_3 - min
        max = np.amax(grey_3)
        grey_3 = grey_3 * 255/max
        grey_3 = np.stack((grey_3,)*3, axis=-1)
        grey_3 = grey_3.astype(np.uint8)
        print(grey_3.shape)
        im = Image.fromarray(grey_3)
        im.save('./maps/'+filename+"_grey_3.png")        



        data = data.astype(np.uint8)
        
        '''
        # ------------------------- Increase image range to 0, 255

        img = img.astype(np.float64)
        for band in range(0, img.shape[2]):
            min = np.amin(img[:,:, band])
            img[:,:, band] = img[:,:, band] - min
            max = np.amax(img[:,:, band])
            img[:,:, band] *= 255.0/max
        img = img.astype(np.uint8)

        # ------------------------- Blur Images

        img = cv2.GaussianBlur(img, (15, 15), 0)
        im = Image.fromarray(img[:, :, 0:3])
        im.save('./maps/'+filename+"_blurred.png")

        # ------------------------- Increase image range to 0, 255
        img = img.astype(np.float32)
        for band in range(0, img.shape[2]):
            min = np.amin(img[:,:, band])
            img[:,:, band] = img[:,:, band] - min
            max = np.amax(img[:,:, band])
            img[:,:, band] *= 255.0/max
            min = np.amin(img[:,:, band])
            max = np.amax(img[:,:, band])
            print(min)
            print(max)
        img = img.astype(np.uint8)


        # ------------------------- Denoise 

        img = img.astype(np.uint8)
        alpha = img[:,:,3]
        alpha2 = img[:,:,4]
        img = cv2.fastNlMeansDenoisingColored(img[:,:,0:3], None, 10, 30, 21, 7)
        alpha = cv2.fastNlMeansDenoising(alpha, None, 30, 21, 7)
        alpha2 = cv2.fastNlMeansDenoising(alpha2, None, 30, 21, 7)

        img= np.insert(img, 3, alpha, axis = 2) # insert b09 band to alpha layer
        img= np.insert(img, 4, alpha2, axis = 2) # insert b09 band to alpha layer
        im = Image.fromarray(img[:, :, 0:3])
        im.save('./maps/'+filename+"_denoised.png")

        # ------------------------- Clustering through K-Means
        img = img.astype(np.float32)
        # re-shape to get 1D array for each layer (a*b, number of bands)
        img = img.reshape((-1, img.shape[2]))

        #print("IMAGE RE-SHAPED: ", img.shape)
        attempts = 10
        K = 12
        ret, label, center = cv2.kmeans(
            img, K, None, None, attempts, cv2.KMEANS_PP_CENTERS)

        center = np.uint8(center)
        print(center)
        res = center[label.flatten()]
        # print("RES: ", res.shape)
        # print(images.reshape((images.shape[1], images.shape[2], 3)).shape)
        img = res.reshape(og_shape)
        im = Image.fromarray(img[:, :, 0:3])
        im.save('./maps/'+filename+"_quantized.png")

        # ------------------------- Edge detecting with Canny
        # using float here since np.nan requires floating points
        '''
        edge = cv2.Canny(img, 2, 5).astype("float")
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
        '''

