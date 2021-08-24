# from glob import glob
import glob
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2  # pip install opencv-contrib-python
from sklearn import preprocessing
from sklearn.cluster import KMeans
from ordered_set import OrderedSet
from skimage.io import imread, imsave
from skimage import exposure
from skimage.exposure import match_histograms

N_OPTICS_BANDS = 4

def generate_histogram(img_name, img, do_print):
    """
    @params: img: can be a grayscale or color image. We calculate the Normalized histogram of this image.
    @params: do_print: if or not print the result histogram
    @return: will return both histogram and the grayscale image 
    """
    if len(img.shape) == 3: # img is colorful, so we convert it to grayscale
        gr_img = np.mean(img, axis=-1)
    else:
        gr_img = img
    '''now we calc grayscale histogram'''
    gr_hist = np.zeros([256])

    for x_pixel in range(gr_img.shape[0]):
        for y_pixel in range(gr_img.shape[1]):
            pixel_value = int(gr_img[x_pixel, y_pixel])
            gr_hist[pixel_value] += 1
            
    '''normalizing the Histogram'''
    gr_hist /= (gr_img.shape[0] * gr_img.shape[1])
    if do_print:
        print_histogram(gr_hist, name=img_name+"n_h_img", title="Normalized Histogram")
    return gr_hist, gr_img
  
def print_histogram(_histrogram, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("hist_" + name)

def equalize_histogram(img, histo, L):
    eq_histo = np.zeros_like(histo)
    en_img = np.zeros_like(img)
    for i in range(len(histo)):
        eq_histo[i] = int((L - 1) * np.sum(histo[0:i]))
    print_histogram(eq_histo, name="eq_"+str(index), title="Equalized Histogram")
    '''enhance image as well:'''
    for x_pixel in range(img.shape[0]):
        for y_pixel in range(img.shape[1]):
            pixel_val = int(img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = eq_histo[pixel_val]
    '''creating new histogram'''
    hist_img, _ = generate_histogram(en_img, print=False, index=index)
    print_img(img=en_img, histo_new=hist_img, histo_old=histo, index=str(index), L=L)
    return eq_histo


def range_0_256(band):
    min = np.amin(band)
    band = band - min
    max = np.amax(band)
    band = band*255/max
    return band

tile_path = "./tiles/accra_-0.2_5.4_0.2_5.6/"

for date_dir_name in os.listdir(tile_path):
    if os.path.isdir(os.path.join(tile_path, date_dir_name)): # ensure is a directory

        date_dir_path = tile_path+date_dir_name+"/" 
        print(date_dir_path)
        # ------------------------- Add Bands Map For Cloud Detection 

        b04 = np.asarray(Image.open(date_dir_path+'l2a_b04.png'), dtype = np.float32)
        b03 = np.asarray(Image.open(date_dir_path+'l2a_b03.png'), dtype = np.float32)
        b02 = np.asarray(Image.open(date_dir_path+'l2a_b02.png'), dtype = np.float32)
        b01 = np.asarray(Image.open(date_dir_path+'l2a_b01.png'), dtype = np.float32)
        b08 = np.asarray(Image.open(date_dir_path+'l1c_b08.png'), dtype = np.float32)
        b09 = np.asarray(Image.open(date_dir_path+'l1c_b09.png'), dtype = np.float32)
        clp = np.asarray(Image.open(date_dir_path+'cloud_prob.png'), dtype = np.float32)
        scl = np.asarray(Image.open(date_dir_path+'l2a_scl.png'), dtype = np.float32)
        '''
        # ------------------------- Get Colour Image
        '''
        rgb_img = b04[:, :, np.newaxis]
        rgb_img = np.insert(rgb_img, rgb_img.shape[2], b03, axis = 2) 
        rgb_img = np.insert(rgb_img, rgb_img.shape[2], b02, axis = 2) 
        rgb_img = rgb_img.astype(np.uint8)
        im = Image.fromarray(rgb_img[:, :, [0,1,2]])
        im.save(date_dir_path+"rgb.png")
        '''
        # ------------------------- Create Composite(s)
        '''
        #composite = range_0_256(b01 * b09)
        composite = range_0_256(b02*b02)


        #composite = b09
        #composite =range_0_256(b01 * b02) # get very white layer
        #composite =range_0_256(composite - b09) # subtract cloud layer
        #composite =range_0_256(composite * b04) # multiply with neutral cloud, bright building layer

        #composite =range_0_256(composite - b09) # subtract cloud layer
        #composite =range_0_256(composite * b04) # multiply with neutral cloud, bright building layer

        #composite =range_0_256(b02 - composite) #
        #composite =range_0_256(composite * b09)

        composite = composite.astype(np.uint8)
        im = Image.fromarray(composite)
        im.save(date_dir_path+"composite.png")

        composite = composite.astype(np.float32)

        img = composite[:, :, np.newaxis]
        #img= np.insert(img, img.shape[2], composite, axis = 2) # insert composite
        #img= np.insert(img, img.shape[2], img_r, axis = 2)  # ignore red

        # ------------------------- Add Bands 

        img = np.insert(img, img.shape[2], b09, axis = 2) 
        img = np.insert(img, img.shape[2], b01, axis = 2) 
        img = np.insert(img, img.shape[2], clp, axis = 2) 
        print("SHAPE: ", img.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB

        # ------------------------- Save Shape

        og_shape = img.shape

        # ------------------------- Increase image range to 0, 255

        for band in range(0, img.shape[2]):
            img[:,:, band] = range_0_256(img[:,:, band])

        # ------------------------- Blur Images

        img = cv2.GaussianBlur(img, (15, 15), 0)
        for band in range(0, img.shape[2]):
            img[:,:, band] = range_0_256(img[:,:, band])

        im = img.astype(np.uint8)
        im = Image.fromarray(im[:, :, [2,1,0]])
        im.save(date_dir_path+"blurred.png")

        # ------------------------- Denoise 

        img = img.astype(np.uint8)
        img2 = cv2.fastNlMeansDenoisingColored(img[:,:,0:3], None, 10, 30, 21, 7)
        if img.shape[2]>=4:
            alpha = cv2.fastNlMeansDenoising(img[:,:,3], None, 30, 21, 7)
            img2= np.insert(img2, 3, alpha, axis = 2) # insert b09 band to alpha layer
        if img.shape[2]>=5:
            alpha = cv2.fastNlMeansDenoising(img[:,:,4], None, 30, 21, 7)
            img2= np.insert(img2, 4, alpha, axis = 2) # insert b09 band to alpha layer
        if img.shape[2]>=6:
            alpha = cv2.fastNlMeansDenoising(img[:,:,5], None, 30, 21, 7)
            img2= np.insert(img2, 5, alpha, axis = 2) # insert b09 band to alpha layer

        img = img2
        img = img.astype(np.float32)

        for band in range(0, img.shape[2]):
            img[:,:, band] = range_0_256(img[:,:, band])

        im = img.astype(np.uint8)
        im = Image.fromarray(im[:, :, [2,1,0]])
        im.save(date_dir_path+"denoised.png")

        # ------------------------- Clustering through K-Means

        # re-shape to get 1D array for each layer (a*b, number of bands)
        img = img.reshape((-1, img.shape[2]))

        print("IMAGE RE-SHAPED: ", img.shape)
        attempts = 10
        K = 12
        ret, label, center = cv2.kmeans(
            img, K, None, None, attempts, cv2.KMEANS_PP_CENTERS)

        center = np.uint8(center)

        res = center[label.flatten()]
        # print("RES: ", res.shape)
        # print(images.reshape((images.shape[1], images.shape[2], 3)).shape)
        img = res.reshape(og_shape)
        img = img.astype(np.float32)

        # ------------------------- Increase image range to 0, 255
        for band in range(0, img.shape[2]):
            img[:,:, band] = range_0_256(img[:,:, band])

        # ------------------------- Save cluster

        grey = img[:, :, 0]
        grey = grey.astype(np.float32)
        shades = OrderedSet([])

        for band in range(1, img.shape[2]):
            grey += img[:, :, band]
        grey = range_0_256(grey/img.shape[2])

        grey = grey.astype(np.uint8)
        for i in range(0, grey.shape[0]):
            for j in range(0, grey.shape[1]):
                shades.add(grey[i,j])

        print(shades)
        im = Image.fromarray(grey)
        im.save(date_dir_path+"clustered.png")

# ------------------------- Mosaicing

# TODO: also swap based on cloud probability, whoever has lowest cloud prob

# Load in cloud mask from each day
scl_imgs = {}
rgb_imgs = {}
cluster_imgs = {}
clp_counts = {}
# Load in cloud mask files and get cloud cover count for each image
for date_dir in os.listdir(tile_path):
    if os.path.isdir(os.path.join(tile_path, date_dir)): # ensure is a directory
        for file_name in os.listdir(tile_path+date_dir+"/"):
            if file_name == "clustered.png":
                #print(tile_path+date_dir+"/l1c_b09.png")
                cluster_imgs[tile_path+date_dir] = np.array(Image.open(tile_path+date_dir+"/clustered.png")) # sub this in for cluster map later
            elif file_name == "cloud_prob.png":
                clp = np.array(Image.open(tile_path+date_dir+"/cloud_prob.png"))
                clp_counts[tile_path+date_dir] = clp.sum() # Count number of pixels that is (probably) a cloud
            elif file_name == "rgb.png":
                #print(tile_path+date_dir+"/rgb.png")
                rgb_imgs[tile_path+date_dir] = np.array(Image.open(tile_path+date_dir+"/rgb.png"))
            elif file_name == "l2a_scl.png":
                #print(tile_path+date_dir+"/l2a_scl.png")
                scl_imgs[tile_path+date_dir] = np.array(Image.open(tile_path+date_dir+"/l2a_scl.png"))


# --------------------------------------------- Get composite SCL
first_key = list(scl_imgs.keys())[0]
composite_scl = scl_imgs[first_key]


for scl in scl_imgs:
    print(scl)
    if scl == first_key:
        continue
    else:                 
        s = scl_imgs[scl] # get scl image
        for row in range(0, composite_scl.shape[0]):
            for col in range(0, composite_scl.shape[1]):
                if s[row, col] == 139:
                    composite_scl[row, col] = 139 # water
                if s[row, col] == 115:
                    composite_scl[row, col] = 115  # land
                if s[row, col] == 92:
                    composite_scl[row, col] = 115  # land

for row in range(0, composite_scl.shape[0]):
    for col in range(0, composite_scl.shape[1]):
        if composite_scl[row, col] == 115 or composite_scl[row, col] == 92: # land or vegetation
            composite_scl[row, col] = 255 # paint land white       
        else:
            composite_scl[row, col] = 0 # paint water/unknown black

Image.fromarray(composite_scl).show() # show composite scl image, which has pixels categorized mainly as water/land
# at this point we have an scl image from all other images (mostly) defining water and land

# np.save("composite_scl", composite_scl)

# --------------------------------------------- Create true cloud masks from cluster maps

cloud_masks = {}

for cl in cluster_imgs:
    c = cluster_imgs[cl] # get a cluster image
    # return a cloud mask (over water)
    for row in range(0, c.shape[0]):
        for col in range(0, c.shape[1]):

            # consider over water
            if composite_scl[row, col] == 255 and c[row, col] > 128:
                c[row, col] = 255
            elif composite_scl[row, col] == 0 and c[row, col] > 12:
                c[row, col] = 255 
            else:
                c[row, col] = 0

    cloud_masks[cl] = c # save cloud mask
#     Image.fromarray(c).show() # show composite scl image, which has pixels categorized mainly as water/land

# ---------------------------------------------  Create true final RGB image

# Choose least cloudy image as base image
least_cloudy_img_path = min(clp_counts, key=clp_counts.get)
print("Least cloud image is: ", least_cloudy_img_path)

composite_rgb = rgb_imgs[least_cloudy_img_path] # base image

# np.save("cloud_masks", cloud_masks)
# np.save("composite_rgb", composite_rgb)
# composite_rgb = np.load("composite_rgb.npy")
# cloud_masks = np.load("cloud_masks.npy", allow_pickle=True)


# ---------------------------------------------  Colour match other images to base image
matched = {}
idx = 0
for rgb_img in rgb_imgs:
    # Match using the right side as reference
    if rgb_img != least_cloudy_img_path:
        rgb_imgs[rgb_img] = match_histograms(rgb_imgs[rgb_img], composite_rgb, multichannel=True)
        imsave(str(idx)+'_result.png', rgb_imgs[rgb_img])
        idx += 1


# Loop through base image and start the mosaicing process!
for i in range(0, cloud_masks[least_cloudy_img_path].shape[0]):
    for j in range(0, cloud_masks[least_cloudy_img_path].shape[1]):
        # Check if there is a cloud at this pixel
        if cloud_masks[least_cloudy_img_path][i,j] == 255:
            # print("There's a cloud here!")
            # See if other tiles have no cloud at same spot
            for date_dir_path in cloud_masks:
                if date_dir_path != least_cloudy_img_path and cloud_masks[date_dir_path][i,j] != 255:
                    # Replace base image's pixel with none cloud pixel
                    composite_rgb[i,j] = rgb_imgs[date_dir_path][i,j]
                    break



Image.fromarray(composite_rgb).show()








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

