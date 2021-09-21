# from glob import glob
import glob
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2  # pip install opencv-contrib-python
import collections
from sklearn import preprocessing
from sklearn.cluster import KMeans
from ordered_set import OrderedSet
from skimage.io import imread, imsave
from skimage import data, exposure
from skimage.exposure import match_histograms

N_OPTICS_BANDS = 4

def plot_histogram(histogram, title):
    plt.figure()
    plt.title(title + " Normalized Histogram")
    #ax = h.add_axes([0,0,1,1])
    range = np.arange(256)
    #plt.ylabel('Number of Pixels')
    #plt.xlabel('Pixel Value')
    #plt.show()
    plt.bar(range, histogram[:,0], width = 1, fc=(1, 0, 0, 0.5), align='center')
    plt.bar(range, histogram[:,1], width = 1, fc=(0, 1, 0, 0.5), align='center')
    plt.bar(range, histogram[:,2], width = 1, fc=(0, 0, 1, 0.5), align='center')
    #h.set_ylim([0, 1])
    plt.show()

def match_histogram(chref, hist, title): 
    # base: cumulative histogram of the reference base map sections of the composite
    # hist: 
    F = np.zeros_like(hist, dtype = float) # create an empty array with same size as base/ref 
    i, j, c, ch = 0, 0, 0, 0
    while ch < 3:
        ch_ref_hist = chref[:, ch]
        ch_hist = hist[:, ch]
        n = np.sum(ch_hist) # get number of pixels contributing to the histo

        r = n/max(ch_ref_hist)
        while i in range(0, 256): # for each brightness value
            if c <= r*ch_ref_hist[j]: # c measures the total number of pixels iterated over
                c = c+ch_hist[i]
                F[i, ch] = j
                print(j)
                i = i+1
            else:
                j = j+1
        i = 0
        j = 0
        c = 0
        ch = ch+1
    plot_histogram(F, title+ " Matched")
    return F

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
        #if date_dir_path != "./tiles/accra_-0.2_5.4_0.2_5.6/2021-08-17/": # uncommenting these two lines will result in only this tile being processed
        #    continue

        # ------------------------- Add Bands Map For Cloud Detection 

        b04 = np.asarray(Image.open(date_dir_path+'l2a_b04.png'), dtype = np.float32)
        b03 = np.asarray(Image.open(date_dir_path+'l2a_b03.png'), dtype = np.float32)
        b02 = np.asarray(Image.open(date_dir_path+'l2a_b02.png'), dtype = np.float32)
        b01 = np.asarray(Image.open(date_dir_path+'l2a_b01.png'), dtype = np.float32)
        b08 = np.asarray(Image.open(date_dir_path+'l1c_b08.png'), dtype = np.float32)
        b09 = np.asarray(Image.open(date_dir_path+'l1c_b09.png'), dtype = np.float32)
        b10 = np.asarray(Image.open(date_dir_path+'l1c_b10.png'), dtype = np.float32)

        clp = np.asarray(Image.open(date_dir_path+'cloud_prob.png'), dtype = np.float32)
        scl = np.asarray(Image.open(date_dir_path+'l2a_scl.png'), dtype = np.float32)

        # ------------------------- Get Colour Image

        rgb_img = b04[:, :, np.newaxis]
        rgb_img = np.insert(rgb_img, rgb_img.shape[2], b03, axis = 2) 
        rgb_img = np.insert(rgb_img, rgb_img.shape[2], b02, axis = 2) 
        rgb_img = rgb_img.astype(np.uint8)
        im = Image.fromarray(rgb_img[:, :, [0,1,2]])
        im.save(date_dir_path+"rgb.png")

        # ------------------------- Create Composite(s)

        max_clp = np.amax(clp) # create a tuple of pixel coords indicating location of highest cloudiness
        min_clp = np.amin(clp) # create a tuple of pixel coords indicating location of lowest cloudiness
        
        # ------------------------- Add Bands 

        #img = clp[:, :, np.newaxis]
        #img = clp
        img = b01
        img = (img * (1+b09/255))
        img = np.clip(img, 0, 255)
        img = img+clp

        #img = b01
        #img = img + b09
        img = img/2
        #img = np.insert(img, img.shape[2], b01, axis = 2) 
        #img = np.insert(img, img.shape[2], b09, axis = 2) 
        img = img.astype(np.uint8)
        #im = Image.fromarray(img).show()

        print("SHAPE: ", img.shape) # make sure shape is (3, a, b) -> (b, a, 3), where 3 is RGB

        # ------------------------- Save Shape

        og_shape = img.shape

        # ------------------------- Denoise 

        img = img.astype(np.uint8)
        img = cv2.fastNlMeansDenoising(img, None, 30, 21, 7)
        '''
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
        '''
        im = Image.fromarray(img)
        #im = Image.fromarray(img[:, :, [2,1,0]])
        im.save(date_dir_path+"denoised.png")

        # ------------------------- Blur Images

        img = cv2.GaussianBlur(img, (15, 15), 0)

        #for band in range(0, img.shape[2]):
        #    img[:,:, band] = range_0_256(img[:,:, band])

        im = img.astype(np.uint8)
        #im = Image.fromarray(im[:, :, [2,1,0]])
        im = Image.fromarray(im)

        im.save(date_dir_path+"clp.png")

        '''
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
        
        #for band in range(0, img.shape[2]):
        #    img[:,:, band] = range_0_256(img[:,:, band])

        # ------------------------- Save cluster

        grey = img[:, :, 0]
        grey = grey.astype(np.float32)
        shades = set() # create a set

        for band in range(1, img.shape[2]):
            grey += img[:, :, band]
        #grey = range_0_256(grey/img.shape[2])
        grey = grey/img.shape[2]

        grey = grey.astype(np.uint8)
        for i in range(0, grey.shape[0]):
            for j in range(0, grey.shape[1]):
                shades.add(grey[i,j])

        print(shades)
        im = Image.fromarray(grey)
        im.save(date_dir_path+"clustered.png")
        '''
# ------------------------- Mosaicing

# TODO: also swap based on cloud probability, whoever has lowest cloud prob

# Load in cloud mask from each day
scl_imgs = {}
rgb_imgs = {}
#cluster_imgs = {}
clp_imgs = {}
clp_counts = {}
histograms = {}
c_histograms = {}
# Load in cloud mask files and get cloud cover count for each image
for date_dir in os.listdir(tile_path):
    if os.path.isdir(os.path.join(tile_path, date_dir)): # ensure is a directory
        for file_name in os.listdir(tile_path+date_dir+"/"):
            if file_name == "clp.png":
                #print(tile_path+date_dir+"/l1c_b09.png")
                clp_imgs[tile_path+date_dir] = np.array(Image.open(tile_path+date_dir+"/clp.png"))

                clp_counts[tile_path+date_dir] = clp_imgs[tile_path+date_dir].sum() # Count number of pixels that is (probably) a cloud
            #if file_name == "clustered.png":
                #print(tile_path+date_dir+"/l1c_b09.png")
                #cluster_imgs[tile_path+date_dir] = np.array(Image.open(tile_path+date_dir+"/clustered.png")) # sub this in for cluster map later
            elif file_name == "rgb.png":
                #print(tile_path+date_dir+"/rgb.png")
                rgb_imgs[tile_path+date_dir] = np.array(Image.open(tile_path+date_dir+"/rgb.png"))
            elif file_name == "l2a_scl.png":
                #print(tile_path+date_dir+"/l2a_scl.png")
                scl_imgs[tile_path+date_dir] = np.array(Image.open(tile_path+date_dir+"/l2a_scl.png"))
            histograms[tile_path+date_dir] = np.zeros((256, 3)) # one for each of R, G, B
            c_histograms[tile_path+date_dir] = np.zeros((256, 3), dtype = float) # one for each of R, G, B


# Choose least cloudy image as base image
#print("CLP COUNTS: ", clp_counts)
clp_counts = {k: v for k, v in sorted(clp_counts.items(), key=lambda item: item[1])} # now sorted in order of least to most cloudy
#print("SORTED CLP COUNTS: ", clp_counts)

 #{k: v for k, v in sorted(clp_counts.items(), key=lambda item: item[1])}
least_cloudy_img_path = min(clp_counts, key=clp_counts.get)
print("Least cloud image is: ", least_cloudy_img_path)
print("Number of clp images: ", len(clp_imgs))


# --------------------------------------------- Create/get composite SCL (water/land classification)

composite_scl = scl_imgs[least_cloudy_img_path] 

if os.path.isfile(tile_path+"scl_composite.png"):
    composite_scl = np.asarray(Image.open(tile_path+"scl_composite.png"), dtype = np.float32)
else:
    composite_bins = np.empty((composite_scl.shape[0], composite_scl.shape[1]), dtype=object)
    for row in range(0, composite_bins.shape[0]):
        for col in range(0, composite_bins.shape[1]):
            composite_bins[row, col] = []
            composite_bins[row, col].append(composite_scl[row, col])

    for scl in scl_imgs:
        print(scl)
        if scl == least_cloudy_img_path:
            continue
        s = scl_imgs[scl] # get scl image
        for row in range(0, s.shape[0]):
            for col in range(0, s.shape[1]):
                if s[row, col] == 139 or s[row, col] == 115 or s[row, col] == 92:
                    composite_bins[row, col].append(s[row, col])

    for row in range(0, composite_bins.shape[0]):
        for col in range(0, composite_bins.shape[1]):
            composite_scl[row, col] = max(composite_bins[row, col], key = composite_bins[row, col].count)
            if composite_scl[row, col] == 139:
                composite_scl[row, col] = 0
            elif composite_scl[row, col] == 115 or composite_scl[row, col] == 92: # land or vegetation
                composite_scl[row, col] = 255 
            else:
                composite_scl[row, col] = 0 # paint unknown as black  

    Image.fromarray(composite_scl).show() # show composite scl image, which has pixels categorized mainly as water/land
    im = Image.fromarray(composite_scl)
    im.save(tile_path+"scl_composite.png")

# at this point we have an scl image from all other images (mostly) defining water and land

# np.save("composite_scl", composite_scl)

# ---------------------------------------------  Create true final RGB image

composite_rgb = rgb_imgs[least_cloudy_img_path].copy() # base rgb image
composite_clp = clp_imgs[least_cloudy_img_path][:,:] # lowest cloud prob image
composite_map = np.zeros(composite_clp.shape, dtype=int) # lowest cloud prob image - fill with zeros indicating base layer

# np.save("cloud_masks", cloud_masks)
# np.save("composite_rgb", composite_rgb)
# composite_rgb = np.load("composite_rgb.npy")
# cloud_masks = np.load("cloud_masks.npy", allow_pickle=True)

# ---------------------------------------------  stack CLPs; get lowest cloud prob pixels and overlay
#Image.fromarray(rgb_imgs[least_cloudy_img_path]).show()

# store histograms of only pixel values that are actually used in final composite
# Loop through base image and start the mosaicing process!
keys = list(clp_counts.keys())
print(keys)
#print(clp_counts)
#print(composite_rgb.shape)
for i in range(0, composite_rgb.shape[0]): 
    for j in range(0, composite_rgb.shape[1]): # for each pixel
        for idx, k in enumerate(clp_counts): # for each clp image in order of lowest cloud probability
            #print(k)
            if k == least_cloudy_img_path:
                continue # skip self
            #print("%d, %d" % (clp_imgs[k][i, j], composite_clp[i, j]))
            if clp_imgs[k][i, j] < composite_clp[i, j]: # Check if there is a cloud at this pixel of the base image
                composite_clp[i, j] = clp_imgs[k][i, j] # fill in the clp composite
                composite_map[i, j] = idx # paint new colour indicative as originating from a map w/ a given index
                composite_rgb[i, j] = rgb_imgs[k][i, j]
                break #try this again after fixing cloud_prob layer detection?

'''
# now count histograms for pixels of composite
for i in range(0, composite_rgb.shape[0]): 
    for j in range(0, composite_rgb.shape[1]): # for each pixel
        m = keys[composite_map[i, j]]
        histograms[m][composite_rgb[i, j, 0], 0] += 1 # add pixel from non composite
        histograms[m][composite_rgb[i, j, 1], 1] += 1
        histograms[m][composite_rgb[i, j, 2], 2] += 1

# at this point we have our true composite img and the individual sections that define the composite. Now create normalized cumulative histograms based on the sections
for h in histograms:
    c_hist = histograms[h].copy()
    for ch in range(0,3):
        sum = 0
        total = np.sum(c_hist[:, ch]) # s should be the same count for all r,g,b histos
        if total != 0:
            for i in range(0, 256):
                sum = sum + c_hist[i, ch]
                c_hist[i, ch] = sum/total # normalization step
    #plot_histogram(c_hist, h)
    c_histograms[h] = c_hist
'''

# hard limit of 255 maps contributing to a single composite (for now)
num_maps = len(clp_counts)
print(num_maps)

composite_map_img = composite_map * 255/num_maps
#composite_map = np.uint8(composite_map)
Image.fromarray(composite_map_img).show()

# count frequency of clp_map values
clp_map_counts = [0] * num_maps
for i in range(0, composite_map.shape[0]): 
    for j in range(0, composite_map.shape[1]): 
        clp_map_counts[composite_map[i,j]] += 1 # find most prevalent maps contributing to composite

# determine base histogram
print(clp_map_counts) 
max_map_count = max(clp_map_counts)
#base_histogram = clp_map_counts.index(max_map_count) #  index of what will be our reference histogram
#print("BASE: ", base_histogram)

#Image.fromarray(composite_rgb).show()
# match other map histogram channels to normalized base histogram channels

'''
for map_idx, h in enumerate(histograms):
    if h != keys[base_histogram]: # for every histogram that isn't the base: match to the base
        #print(histograms[keys[base_histogram]])
        F = match_histogram(c_histograms[keys[base_histogram]], histograms[h], h) # pass in c_hist of reference, hist of img to be modded
        print("MATCHED")
        for ch in range(0, composite_rgb.shape[2]):
            for i in range(0, composite_rgb.shape[0]):
                for j in range(0, composite_rgb.shape[1]):
                    if composite_map[i, j] == map_idx: # Manipulate only the pixels belonging to a non-base map; change all pixels with value h to value b
                            composite_rgb[i, j, ch] = F[composite_rgb[i, j, ch], ch]
        print("COMPOSED")

        Image.fromarray(composite_rgb).show()

'''

#print(histograms[least_cloudy_img_path])

composite_rgb = composite_rgb.astype(np.float32)
for band in range(0, composite_rgb.shape[2]):
    composite_rgb[:,:,band] = range_0_256(composite_rgb[:,:,band])
composite_rgb = composite_rgb.astype(np.uint8)

Image.fromarray(composite_clp).show()
rgb_img = Image.fromarray(composite_rgb[:,:,[0,1,2]])
rgb_img.show()


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

