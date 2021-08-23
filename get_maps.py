# import rasterio
# import geopyspark as gps
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from datetime import datetime
import dateutil.relativedelta
# from pyspark import SparkContext
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, WmsRequest, WcsRequest, MimeType, CRS, BBox, DataCollection, MimeType, SentinelHubRequest, SentinelHubDownloadClient, bbox_to_dimensions, DownloadRequest
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest, get_s2_evalscript
from PIL import Image, ImageEnhance
from scipy.ndimage.filters import gaussian_filter
import time



# b02 red b03 green b04 blue ; CLD = sentinel2 cloud detection
evalscript_l1c = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "dataMask"],
                units: "reflectance"
            }],
            output: {
                bands: 14,
                sampleType: "FLOAT32",
            },
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12, sample.dataMask];
    }
"""
# bands: ["B02", "B03", "B04", "B08", "SCL", "CLP", "CLM"],
# return [sample.B02*255, sample.B03*255, sample.B04*255, sample.B08*255, sample.SCL, sample.CLP, sample.CLM];
#mosaicking: "ORBIT"

def get_l1c_request(time_interval):
    return SentinelHubRequest(
        evalscript=evalscript_l1c,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C, 
                time_interval=time_interval,
                #maxcc = 0.4,
                 # if below is commented out, then most recent first
                mosaicking_order='leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response(
                'default', MimeType.TIFF)
        ],
        bbox=betsiboka_bbox,
        size=betsiboka_size,
        config=config
    )


evalscript_l2a = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03", "B04", "B08", "B8A", "B09", "B11", "B12", "CLP", "SCL"],
            }],
            output: {
                bands: 11,
                sampleType: "FLOAT32",
            },
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12, sample.CLP, sample.SCL];
    }
"""

def get_l2a_request(time_interval):
    return SentinelHubRequest(
        evalscript=evalscript_l2a,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A, # L2A atmospheric corrected data
                time_interval=time_interval,
                #maxcc = 0.4,
                 # if below is commented out, then most recent first
                mosaicking_order='leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response(
                'default', MimeType.TIFF)
        ],
        bbox=betsiboka_bbox,
        size=betsiboka_size,
        config=config
    )


def plot_image(image=None, mask=None, ax=None, factor=3.5/255, clip_range=(0, 1), **kwargs):
    """ Utility function for plotting RGB images and masks.
    """
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    mask_color = [255, 255, 255, 255] if image is None else [255, 255, 0, 100]

    if image is None:
        if mask is None:
            raise ValueError('image or mask should be given')
        image = np.zeros(mask.shape + (3,), dtype=np.uint8)

    ax.imshow(np.clip(image * factor, *clip_range), **kwargs)

    if mask is not None:
        cloud_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        cloud_image[mask == 1] = np.asarray(mask_color, dtype=np.uint8)

        ax.imshow(cloud_image)


def plot_probabilities(image, proba, factor=3.5/255):
    """ Utility function for plotting a RGB image and its cloud probability map next to each other.
    """
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 2, 1)
    ax.imshow(np.clip(image * factor, 0, 1))
    ax = plt.subplot(1, 2, 2)
    ax.imshow(proba, cmap=plt.cm.inferno)

# return band scaled between range 0, 255
def get_scaled_band(idx, map):
    b = map[:, :, idx]*255
    min = np.amin(b)
    b = b - min
    max = np.amax(b)
    b = b*255/max
    return b

# sign in to sentinelhub
config = SHConfig()

#config.sh_client_id = '5c047aa4-a418-44d5-8439-9961b84e1d1a'
#config.sh_client_secret = '+eHOXYc&5.rTUKN_Gb]?OS4jr/xu?_5mtMMUMqin'

#leo5757577@hotmail.com
#uJSxW5TD2xPyE3z
config.sh_client_id = '3b804ca0-0b07-4944-8d57-370037a191a7'
config.sh_client_secret = 'k56Pk,hBQubcKwJqr@qvQITX)EFZltq{i.0rM|@/'

if config.instance_id == '':
    print("Warning! To use OGC functionality of Sentinel Hub, please configure the `instance_id`.")
config.save()

resolution = 10

# longitude + latitude or lower left, upper right corners

sites = {}
# coast of hanoi:
# sites['hanoi'] = [106.00, 13.00, 116.00, 22.50]
sites['accra'] = [-0.20, 5.40, 0.20, 5.60]
# sites['example'] = [-90.9217, 14.4191, -90.8187, 14.5520]

DEBUG_FILE_OPERATION = "[File_OPERATION] "
DEBUG_TILE_QUERY = "[TILE_QUERY] "
DEBUG_POSSIBLE_ERR = "[POSSIBLE_ERR]"
DEBUG_CLOUD_DETECTION = "[CLOUD_DETECTION] "

if not os.path.exists('./tiles/'): # Create directory tiles will be saved in
    print(DEBUG_FILE_OPERATION+"Tile directory does not exist, creating one now.")
    os.mkdir('./tiles/')

for s in sites:

    folder_path = './tiles/' + s
    for digit in sites[s]:
        folder_path += "_" + str(digit)
    # print("Folder Path for " + s + " : ", folder_path)
    if not os.path.exists(folder_path):
        print(DEBUG_FILE_OPERATION + "Folder does not exist: " + folder_path + ". Creating one now.")
        os.mkdir(folder_path)
    else:
        print(DEBUG_FILE_OPERATION + "Folder exists for tile at:" + folder_path + ".")

    start_x = sites[s][0]
    end_x = sites[s][2]
    start_y = sites[s][1]
    end_y = sites[s][3]
    x = start_x
    y = start_y
    x_delta = 0.20
    y_delta = 0.20
    if(end_x < start_x):
        x_delta = -0.20
    if(end_y < start_y):
        y_delta = -0.20

    # while (x < end_x):
    #     while (y < end_y):
    #         folder_path = './maps/'+s + '/' + str(x) + '_'+str(y)
    #         if not os.path.exists(folder_path):
    coords = [x, y, x+x_delta, y+y_delta]
    # betsiboka_coords_wgs84 = [46.15, -16.20, 46.35, -16.00]  # 0.2 x0.2 box
    betsiboka_bbox = BBox(bbox=coords, crs=CRS.WGS84)
    betsiboka_size = bbox_to_dimensions(
        betsiboka_bbox, resolution=resolution)
    """All requests require bounding box to be given as an instance of sentinelhub.geometry.BBox with corresponding Coordinate Reference System (sentinelhub.geometry.CRS)"""
    # get one month of data
    today = datetime.today()
    interval_length = 7 # 7 days
    earliest_date = today + dateutil.relativedelta.relativedelta(weeks=-8) # get two months of data
    last_week = today + dateutil.relativedelta.relativedelta(days=-interval_length) # one week before today
    slots = [] # all weeks of data to be queried
    while today > earliest_date:
        today_str = today.strftime('%Y-%m-%d')  # get in string format
        last_week_str = last_week.strftime('%Y-%m-%d')
        slots.append((last_week_str, today_str)) # 1 week intervals
        today = last_week
        last_week = today + dateutil.relativedelta.relativedelta(days=-interval_length)
    print(DEBUG_TILE_QUERY, len(slots), " tiles to download for ", s, sites[s]) # should have a number of time intervals
    
    l2a_list_of_requests = [get_l2a_request(slot) for slot in slots]
    l2a_list_of_requests = [request.download_list[0] for request in l2a_list_of_requests] # now have a list of requests
    l1c_list_of_requests = [get_l1c_request(slot) for slot in slots]
    l1c_list_of_requests = [request.download_list[0] for request in l1c_list_of_requests] 

    l2a_raw_maps_file_names = ["l2a_b01.png", "l2a_b02.png","l2a_b03.png","l2a_b04.png", "l2a_b08.png", "l2a_b8a.png", "l2a_b09.png", "l2a_b11.png", "l2a_b12.png", "l2a_clp.png", "l2a_scl.png"]
    l2a_raw_maps = []
    if not os.path.exists(folder_path+"/l2a_raw_maps.npy"):
        print(DEBUG_TILE_QUERY+" Beginning l2a_raw_maps download for ", s, sites[s])
        l2a_raw_maps = np.array(SentinelHubDownloadClient(config=config).download(l2a_list_of_requests, max_threads=5), dtype= np.float32)
        np.save(folder_path+"/l2a_raw_maps", l2a_raw_maps)
        print(DEBUG_TILE_QUERY+" Finished l2a_raw_maps download for ", s, sites[s])
    else:
        print(DEBUG_TILE_QUERY+" l2a_raw_maps loaded for ", s, sites[s])
        np.load(folder_path+"/l2a_raw_maps.npy")

    # Create directory for the different days with picture of each band
    for idx, map in enumerate(l2a_raw_maps):
        for band, f in enumerate(l2a_raw_maps_file_names):
            if not os.path.exists(folder_path+"/"+slots[idx][1]):
                Path(folder_path+"/"+slots[idx][1]+"/").mkdir(parents=True, exist_ok=True)
            if not os.path.exists(folder_path+"/"+slots[idx][1]+"/"+f):
                b = map[:, :, band]
                if f != "l2a_scl.png":
                    b = get_scaled_band(band, map)
                else:
                    b *= 255/11
                b = b.astype(np.uint8)
                im = Image.fromarray(b)
                im.save(folder_path+"/"+slots[idx][1]+"/"+f)

    l1c_raw_maps_file_names = ["l1c_b01.png","l1c_b02.png","l1c_b03.png","l1c_b04.png","l1c_b05.png","l1c_b06.png","l1c_b07.png","l1c_b08.png","l1c_b8a.png","l1c_b09.png","l1c_b10.png","l1c_b11.png","l1c_b12.png"]
    l1c_raw_maps = []
    if not os.path.exists(folder_path+"/l1c_raw_maps.npy"):
        print(DEBUG_TILE_QUERY+" Beginning l1c_raw_maps download for ", s, sites[s])
        l1c_raw_maps = np.array(SentinelHubDownloadClient(config=config).download(l1c_list_of_requests, max_threads=5), dtype= np.float32)
        np.save(folder_path+"/l1c_raw_maps", l1c_raw_maps)
        print(DEBUG_TILE_QUERY+" Finished l1c_raw_maps download for ", s, sites[s])
    else:
        print(DEBUG_TILE_QUERY+" l1c_raw_maps loaded for ", s, sites[s])
        np.load(folder_path+"/l1c_raw_maps.npy")

    
    for idx, map in enumerate(l1c_raw_maps):
        for band, f in enumerate(l1c_raw_maps_file_names):
            if not os.path.exists(folder_path+"/"+slots[idx][1]):
                Path(folder_path+"/"+slots[idx][1]+"/").mkdir(parents=True, exist_ok=True)
            if not os.path.exists(folder_path+"/"+slots[idx][1]+"/"+f):
                b = get_scaled_band(band, map)
                b = b.astype(np.uint8)
                im = Image.fromarray(b)
                im.save(folder_path+"/"+slots[idx][1]+"/"+f)

    # If downloaded tile bad
    # is_all_zero = np.all(l1c_raw_maps == 0)
    # if is_all_zero:
    #     print(DEBUG_POSSIBLE_ERR+" Raw L1C maps contains only 0 for "+s,sites[s])
    # else:
    #     print(DEBUG_POSSIBLE_ERR+" Raw L1C maps has non-zero items for "+s,sites[s])

    # print("Raw maps: ", l1c_raw_maps)
    # print("Raw rgb maps shape: ", l2a_raw_maps.shape)  # should be like (3, 13259,1399, 5)
    # print("Raw rgb maps data type: ", l2a_raw_maps.dtype) # float32 for best results in cloud detection
    # print("Raw maps shape: ", l1c_raw_maps.shape)  # should be like (3, 13259,1399, 5)
    # print("Raw maps data type: ", l1c_raw_maps.dtype) # float32 for best results in cloud detection

    # is_all_zero = np.all(raw_maps == 0)
    # if is_all_zero:
    #     print('Raw L1C maps contains only 0 for '+s,sites[s])
    # else:
    #     print('Raw L1C maps has non-zero items for '+s,sites[s])

    for idx, l1c_map in enumerate(l1c_raw_maps):

        '''
        # Brighten image for viewing purposes
        for band in range(0, l1c_map.shape[2]):
            l1c_map[:, :, band] = get_scaled_band(band, l1c_map)
        '''
        # ------------------ Cloud Detection Begins

        # none rgb images
        bands = l1c_map[..., :-1]
        mask = l1c_map[..., -1]

        # print("bands: ", bands)
        # print("mask: ", mask)
        # print("Band shape, mask shape: ", bands.shape, mask.shape)

        cloud_detector = S2PixelCloudDetector(
            threshold=0.15,
            average_over=5,
            dilation_size=1,
            all_bands=True
        )

        # Cloud prob data saving
        cloud_prob = []
        if not os.path.exists(folder_path+"/"+slots[idx][1]+"/cloud_prob.png"):
            print(DEBUG_CLOUD_DETECTION+" Running cloud probability detector for "+s,sites[s]," pic ", idx)
            start_time = time.time()
            cloud_prob = cloud_detector.get_cloud_probability_maps(bands)
            print(DEBUG_CLOUD_DETECTION+" Cloud probability detector took ", time.time() - start_time," for "+s,sites[s]," pic ", idx)
            cloud_prob *= 255
            cloud_prob = cloud_prob.astype(np.uint8)
            im = Image.fromarray(cloud_prob)
            im.save(folder_path+"/"+slots[idx][1]+"/cloud_prob.png")

        # Cloud mask data saving
        cloud_mask = []
        if not os.path.exists(folder_path+"/"+slots[idx][1]+"/cloud_mask.png"): 
            print(DEBUG_CLOUD_DETECTION+"Running cloud mask detector for "+s,sites[s]," pic ",idx)
            start_time = time.time()
            cloud_mask = cloud_detector.get_cloud_masks(bands)
            print(DEBUG_CLOUD_DETECTION+" Cloud mask detector took ", time.time() - start_time," for "+s,sites[s]," pic ",idx)
            cloud_mask *= 255
            cloud_mask = cloud_mask.astype(np.uint8)            
            im = Image.fromarray(cloud_mask)
            im.save(folder_path+"/"+slots[idx][1]+"/cloud_mask.png")

        # Check Cloud Prob data
        is_all_zero = np.all((cloud_prob == 0))
        if is_all_zero:
            print(DEBUG_CLOUD_DETECTION+' Cloud prob contains only 0 for '+s,sites[s])
        else:
            print(DEBUG_CLOUD_DETECTION+' Cloud prob has non-zero items for '+s,sites[s])
        # plot_probabilities(rgb_map, cloud_prob)

        # Check Cloud Mask data
        is_all_zero = np.all((cloud_mask == 0))
        if is_all_zero:
            print('Cloud mask contains only 0')
        else:
            print('Cloud mask has non-zero items')
        #plot_image(image=l1c_map[:,:, [3,2,1]], mask=cloud_mask)

        print("max infr: ", np.amax(l1c_map[:, :, 7]))
        print("min infr: ", np.amin(l1c_map[:, :, 7]))
        
        # bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "dataMask"],

        '''
        b = get_scaled_band(1, l1c_map) 
        img_b = Image.fromarray(b, 'L')
        img_b.show()
        '''

        # print("cloud probability: ", cloud_prob)
        # print("cloud probability shape: ", cloud_prob.shape)


        # print("cloud mask: ", cloud_mask)
        # img_mask = Image.fromarray(cloud_mask, 'L')
        # img_mask.show()


        # plt.show() # need to close all open windows before progressing to next one

        
        # cloud_prob*=255
        # cloud_prob = cloud_prob.astype(np.uint8)
        # for i in range(0, cloud_prob.shape[0]):
        #     for j in range(0, cloud_prob.shape[1]):
        #         if cloud_prob[i, j]/255 >= 0.3:
        #             cloud_prob[i, j] = 225
        #         else:
        #             cloud_prob[i, j] = 0
        
        
        # img_prob = Image.fromarray(cloud_prob, 'L')
        # img_prob.show()