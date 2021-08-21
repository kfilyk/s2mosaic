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



tile_path = "./tiles/accra_-0.2_5.4_0.2_5.6/"

for date_dir_name in os.listdir(tile_path):
    if os.path.isdir(os.path.join(tile_path, date_dir_name)): # ensure is a directory

        # files = []
        # filelist = []
        '''
        for dirpath, dirnames, filenames in os.walk("./maps/accra"):
        for filename in [f for f in filenames if f.endswith(".png")]:
            filelist.append(os.path.join(dirpath,filename))
            files.append(filename)
        print(files)
        print('\n')
        print(filelist)
        '''
        # date_dir_path = tile_path+"/"+date_dir_name+"/" # file path of 
        # filelist.append('./maps/accra/l1c_rgb.png')

        # ------------------------- Add CLP Map For Cloud Detection 

        # r = Image.open(date_dir_path+'l2a_b02.png')
        # g = Image.open(date_dir_path+'l2a_b03.png')
        # b = Image.open(date_dir_path+'l2a_b04.png')

        # img = np.asarray(r)
        # img_g = np.asarray(g)
        # img_b = np.asarray(b)
        # img = img[:, :, np.newaxis]
        # img= np.insert(img, img.shape[2], img_g, axis = 2) # insert b09 band to alpha layer
        # img= np.insert(img, img.shape[2], img_b, axis = 2) # insert b09 band to alpha layer

        # print("IMG SHAPE: ", img.shape)

        # im = Image.fromarray(img[:, :, [2,1,0]])
        # im.save(date_dir_path+"_rgb.png")
        # im.show()

        # ------------------------- Add CLP Map For Cloud Detection 

        # b_image = Image.open(date_dir_path+'cloud_prob.png')
        # b = np.asarray(b_image, dtype = np.float32)
        # img= np.insert(img, img.shape[2], b, axis = 2) # insert b09 band to alpha layer
        # print("SHAPE: ", img.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB


        # # ------------------------- Add B09 Band For Cloud Detection 

        # b_image = Image.open(date_dir_path+'l1c_b09.png')
        # b = np.asarray(b_image, dtype = np.float32)
        # img= np.insert(img, img.shape[2], b, axis = 2) # insert b09 band to alpha layer
        # print("SHAPE: ", img.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB

        # '''
        # # ------------------------- Add B01 Band For Cloud Detection 

        # b_image = Image.open('./maps/accra/b01.png')
        # b = np.asarray(b_image, dtype = np.float32)
        # img= np.insert(img, img.shape[2], b, axis = 2) # insert b09 band to alpha layer
        # print("SHAPE: ", img.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB
        # '''

        # # ------------------------- Save Shape

        # og_shape = img.shape

        # # ------------------------- Add B01 Band For Cloud Detection 

        # '''
        # aerosol = Image.open('./maps/accra/accra_aerosol.png')
        # a_data = np.asarray(aerosol, dtype = np.float32)
        # print("AERO SHAPE: ", a_data.shape)
        # """
        # min = np.amin(a_data) 
        # a_data = a_data - min
        # max = np.amax(a_data)
        # a_data = a_data * 255/max
        # """
        # a_data = a_data.astype(np.uint8)

        # im = Image.fromarray(a_data)
        # im.save('./maps/'+"_aero.png")        
        # a_data = a_data.astype(np.float64)
        # '''

        # # ------------------------- Increase image range to 0, 255

        # img = img.astype(np.float64)
        # for band in range(0, img.shape[2]):
        #     min = np.amin(img[:,:, band])
        #     img[:,:, band] = img[:,:, band] - min
        #     max = np.amax(img[:,:, band])
        #     img[:,:, band] *= 255.0/max
        # img = img.astype(np.uint8)

        # # ------------------------- Blur Images

        # img = cv2.GaussianBlur(img, (15, 15), 0)
        # im = Image.fromarray(img[:, :, [2,1,0]])
        # im.save(date_dir_path+"_blurred.png")

        # # ------------------------- Increase image range to 0, 255
        # img = img.astype(np.float32)
        # for band in range(0, img.shape[2]):
        #     min = np.amin(img[:,:, band])
        #     img[:,:, band] = img[:,:, band] - min
        #     max = np.amax(img[:,:, band])
        #     img[:,:, band] *= 255.0/max
        #     min = np.amin(img[:,:, band])
        #     max = np.amax(img[:,:, band])
        #     print(min)
        #     print(max)
        # img = img.astype(np.uint8)


        # # ------------------------- Denoise 

        # img = img.astype(np.uint8)
        # img2 = cv2.fastNlMeansDenoisingColored(img[:,:,0:3], None, 10, 30, 21, 7)
        # if img.shape[2]>=4:
        #     alpha = cv2.fastNlMeansDenoising(img[:,:,3], None, 30, 21, 7)
        #     img2= np.insert(img2, 3, alpha, axis = 2) # insert b09 band to alpha layer
        # if img.shape[2]>=5:
        #     alpha = cv2.fastNlMeansDenoising(img[:,:,4], None, 30, 21, 7)
        #     img2= np.insert(img2, 4, alpha, axis = 2) # insert b09 band to alpha layer
        # if img.shape[2]>=6:
        #     alpha = cv2.fastNlMeansDenoising(img[:,:,5], None, 30, 21, 7)
        #     img2= np.insert(img2, 5, alpha, axis = 2) # insert b09 band to alpha layer

        # img = img2
        # print("IMAGE SHAPE: ", img.shape)

        # im = Image.fromarray(img[:, :, [2,1,0]])
        # im.save(date_dir_path+"_denoised.png")

        # # ------------------------- Increase image range to 0, 255

        # img = img.astype(np.float64)
        # for band in range(0, img.shape[2]):
        #     min = np.amin(img[:,:, band])
        #     img[:,:, band] = img[:,:, band] - min
        #     max = np.amax(img[:,:, band])
        #     img[:,:, band] *= 255.0/max
        # img = img.astype(np.uint8)

        # im = Image.fromarray(img[:, :, [2,1,0]])
        # im.save(date_dir_path+"_denoised.png")

        # # ------------------------- Clustering through K-Means

        # img = img.astype(np.float32)
        # # re-shape to get 1D array for each layer (a*b, number of bands)
        # img = img.reshape((-1, img.shape[2]))

        # print("IMAGE RE-SHAPED: ", img.shape)
        # attempts = 10
        # K = 12
        # ret, label, center = cv2.kmeans(
        #     img, K, None, None, attempts, cv2.KMEANS_PP_CENTERS)

        # center = np.uint8(center)
        # print(center)
        # res = center[label.flatten()]
        # # print("RES: ", res.shape)
        # # print(images.reshape((images.shape[1], images.shape[2], 3)).shape)
        # img = res.reshape(og_shape)
        # im = Image.fromarray(img[:, :, [2, 1, 0]]) # img[:, :, [4, 3, 0]]
        # im.save(date_dir_path+"_clustered.png")



        # ------------------------- Mosaicing


        # Load in cloud mask from each day
        # rgb_imgs = {}
        # cloud_masks = {}
        # cloud_masks_cloud_count = {}

        # # Load in cloud mask files and get cloud cover count for each image
        # for cloud_mask_file in os.listdir(tile_path+date_dir_name+"/"):
        #     if cloud_mask_file == "cloud_mask.png":
        #         # print(dsirectory_path+"/"+date_dir_name+"/cloud_mask.png")
        #         rgb_imgs[date_dir_name] = np.asarray(Image.open(tile_path+date_dir_name+"/"+"_rgb.png"))

        #         Image.open(tile_path+date_dir_name+"/"+"_rgb.png").show()

        #         cloud_masks[date_dir_name] = np.asarray(Image.open(tile_path+date_dir_name+"/cloud_mask.png"))
        #         cloud_masks_cloud_count[date_dir_name] = np.sum(np.asarray(Image.open(tile_path+date_dir_name+"/cloud_mask.png")) % 2 == 1) # Sum up 1's

        # # Choose least cloudy image as base image
        # least_cloudy_img_dir = min(cloud_masks_cloud_count, key=cloud_masks_cloud_count.get)
        # print(least_cloudy_img_dir)



        # Loop through base image and start the mosaicing process!
        # for i in range(0, cloud_masks[least_cloudy_img_dir].shape[0]):
        #     for j in range(0, cloud_mask[least_cloudy_img_dir].shape[1]):










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
        im.save("./maps/"+"_edge_overlay.png")
        '''

