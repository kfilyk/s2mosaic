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
evalscript_10m_bands = """
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

def get_map_request(time_interval):
    return SentinelHubRequest(
        evalscript=evalscript_10m_bands,
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


evalscript_RGB_band = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"],
            }],
            output: {
                bands: 3,
                sampleType: "UINT8",
            },
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02*255, sample.B03*255, sample.B04*255];
    }
"""

def get_rgb_map_request(time_interval):
    return SentinelHubRequest(
        evalscript=evalscript_RGB_band,
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




# sign in to sentinelhub
config = SHConfig()

config.sh_client_id = '5c047aa4-a418-44d5-8439-9961b84e1d1a'
config.sh_client_secret = '+eHOXYc&5.rTUKN_Gb]?OS4jr/xu?_5mtMMUMqin'
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
DEBUG_TILE_OPERATION = "[TILE_OPERATION] "
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
    earliest_date = today + dateutil.relativedelta.relativedelta(weeks=-3)
    last_week = today + dateutil.relativedelta.relativedelta(days=-interval_length) # one week before today
    slots = [] # all weeks of data to be queried
    while today > earliest_date:
        today_str = today.strftime('%Y-%m-%d')  # get in string format
        last_week_str = last_week.strftime('%Y-%m-%d')
        slots.append((last_week_str, today_str)) # 1 week intervals
        today = last_week
        last_week = today + dateutil.relativedelta.relativedelta(days=-interval_length)
    print(DEBUG_TILE_QUERY, len(slots), " tiles to download for ", s, sites[s]) # should have a number of time intervals
    
    rgb_list_of_requests = [get_rgb_map_request(slot) for slot in slots]
    rgb_list_of_requests = [request.download_list[0] for request in rgb_list_of_requests] # now have a list of requests

    list_of_requests = [get_map_request(slot) for slot in slots]
    list_of_requests = [request.download_list[0] for request in list_of_requests] # now have a list of 

    # raw_maps = np.array(map_requests.get_data(), dtype=np.float64) # get array of all maps - note that SCL cloud cover map
    # print("raw_maps shape: ", raw_maps.shape)  # should be like (3, 13259,1399, 5)
    # print("raw_maps data type: ", raw_maps.dtype)
    # maps = np.copy(raw_maps)
    # average_map = np.zeros_like(maps[0, :, :, :4]) # create a single map instance of size (w, h, 4) instead of 5
    # pixel_count = np.zeros_like(maps[0, :, :]) # counts how many different sub map values have been accumulated for a given pixel location
    # print(average_map.shape)  # should be like (3, 13259,1399, 5)
    # print(pixel_count.shape)  # should be like (3, 13259,1399, 5)]


    # Download if rgb raw maps exist, load if not.
    rgb_raw_maps_file_name = "rgb_raw_maps.npy"
    rgb_raw_maps = []
    if not os.path.exists(folder_path+"/"+rgb_raw_maps_file_name):
        print(DEBUG_FILE_OPERATION+rgb_raw_maps_file_name+" file does not exist for ",s,sites[s]," downloading...")
        # Download maps
        rgb_raw_maps = np.array(SentinelHubDownloadClient(config=config).download(rgb_list_of_requests, max_threads=5), dtype= np.float32)
        # Save raw maps
        print(DEBUG_TILE_QUERY+" Successfully downloaded rgb raw map tiles for: ",s, sites[s])
        np.save(folder_path+"/"+rgb_raw_maps_file_name, rgb_raw_maps)
    else:
        # Load saved maps
        rgb_raw_maps = np.load(folder_path+"/"+rgb_raw_maps_file_name)
        print(DEBUG_TILE_QUERY+" Loading in "+rgb_raw_maps_file_name+" for "+ s,sites[s])

    # Download if raw maps exist, load if not.
    raw_maps_file_name = "raw_maps.npy"
    raw_maps = []
    if not os.path.exists(folder_path + "/" + raw_maps_file_name):
        print(DEBUG_FILE_OPERATION+raw_maps_file_name+" file does not exist for ",s,sites[s]," downloading...")
        # Download maps
        raw_maps = np.array(SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5), dtype= np.float32)
        # Save raw maps
        print(DEBUG_TILE_QUERY+" Downloaded raw map tiles for ",s,sites[s])
        np.save(folder_path+"/"+raw_maps_file_name, raw_maps)
    else:
        # Load saved maps
        raw_maps = np.load(folder_path + "/" + raw_maps_file_name)
        print(DEBUG_TILE_QUERY+" Loading in "+ raw_maps_file_name+" for "+s,sites[s])

    # # If downloaded tile bad
    # is_all_zero = np.all(rgb_raw_maps == 0)
    # if is_all_zero:
    #     print(DEBUG_POSSIBLE_ERR+" Raw rgb maps contains only 0 for "+s,sites[s])
    # else:
    #     print(DEBUG_TILE_OPERATION+" Raw rgb maps has non-zero items for "+s,sites[s])

    # print("Raw maps: ", raw_maps)
    # print("Raw rgb maps shape: ", rgb_raw_maps.shape)  # should be like (3, 13259,1399, 5)
    # print("Raw rgb maps data type: ", rgb_raw_maps.dtype) # float32 for best results in cloud detection
    # print("Raw maps shape: ", raw_maps.shape)  # should be like (3, 13259,1399, 5)
    # print("Raw maps data type: ", raw_maps.dtype) # float32 for best results in cloud detection

    # is_all_zero = np.all(raw_maps == 0)
    # if is_all_zero:
    #     print('Raw maps contains only 0')
    # else:
    #     print('Raw maps has non-zero items')

    for m in range(0, len(rgb_raw_maps)):

        # RGB image
        rgb_map = rgb_raw_maps[m]

        # Brighten image for viewing purposes
        # rgb_map = rgb_map.astype(np.float64)
        # for band in range(0, 3):
        #     # print("band shape: ", rgb_map[:, :, band].shape)
        #     min = np.amin(rgb_map[:, :, band])
        #     # print("min: ", min)
        #     rgb_map[:,:, band] = rgb_map[:,:,band]-min
        #     max = np.amax(rgb_map[:,:,band])
        #     # print("max: ", max)
        #     rgb_map[:, :, band] *= (255.0/float(max))
        #     # print("min: ", np.amin(rgb_map[:, :, band]))
        #     # print("max: ", np.amax(rgb_map[:, :, band]))
        # rgb_map = rgb_map.astype(np.uint8)

        # img_rgb = Image.fromarray(rgb_map[:, :, [2,1,0]], 'RGB')
        # img_rgb.show()
        
        # The ghetto way to display and brighten image for funs
        # for_native_preview_rgb_map = np.dstack([rgb_map[:, :, 2],rgb_map[:, :, 1],rgb_map[:, :, 0]])*2
        # img_rgb = Image.fromarray(for_native_preview_rgb_map, 'RGB')
        


        # ------------------ Cloud Detection Begins

        # none rgb images
        the_map = raw_maps[m]

        bands = the_map[..., :-1]
        mask = the_map[..., -1]

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
        if not os.path.exists(folder_path+"/cloud_prob-"+str(m)+".npy"):
            print(DEBUG_CLOUD_DETECTION+" Running cloud probability detector for "+s,sites[s]," pic ", m)
            start_time = time.time()
            cloud_prob = cloud_detector.get_cloud_probability_maps(bands)
            print(DEBUG_CLOUD_DETECTION+" Cloud probability detector took ", time.time() - start_time," for "+s,sites[s]," pic ", m)
            np.save(folder_path+"/cloud_prob-"+str(m), cloud_prob)
        else:
            cloud_prob = np.load(folder_path+"/cloud_prob-"+str(m)+".npy")

        # Cloud mask data saving
        cloud_mask = []
        if not os.path.exists(folder_path+"/cloud_mask-"+str(m)+".npy"): 
            print(DEBUG_CLOUD_DETECTION+"Running cloud mask detector for "+s,sites[s]," pic ",m)
            start_time = time.time()
            cloud_mask = cloud_detector.get_cloud_masks(bands)
            print(DEBUG_CLOUD_DETECTION+" Cloud mask detector took ", time.time() - start_time," for "+s,sites[s]," pic ",m)
            np.save(folder_path+"/cloud_mask-"+str(m), cloud_mask)
        else:
            cloud_mask = np.load(folder_path+"/cloud_mask-"+str(m)+".npy")

        # print("Cloud prob: ", cloud_prob)
        # print("Cloud mask:  ", cloud_mask)
        # print("Cloud prob shape: ", cloud_prob.shape)
        # print("Cloud mask shape: ", cloud_mask.shape)

        # Cloud prob not working properly
        is_all_zero = np.all((cloud_prob == 0))
        if is_all_zero:
            print('Cloud prob contains only 0')
        else:
            print('Cloud prob has non-zero items')
        # plot_probabilities(rgb_map, cloud_prob)

        # Mask working properly
        is_all_zero = np.all((cloud_mask == 0))
        if is_all_zero:
            print('Cloud mask contains only 0')
        else:
            print('Cloud mask has non-zero items')
        plot_image(image=rgb_map, mask=cloud_mask)

        print("max infr: ", np.amax(the_map[:, :, 7]))
        print("min infr: ", np.amin(the_map[:, :, 7]))
        
        
        min = np.amin(the_map[:, :, 0])
        the_map[:, :, 0] = the_map[:, :, 0] - min
        max = np.amax(the_map[:, :, 0])
        # the_map[:, :, 0] *=255/max

        '''
        # Combined
        combined_img = np.empty(shape=(the_map[:, :, 0].shape[0],the_map[:, :, 0].shape[1]))
        for i in range(0, the_map[:, :, 0].shape[0]):
            for j in range(0, the_map[:, :, 0].shape[1]):
                combined_img[i,j] = (the_map[i, j, 0] + the_map[i, j, 1] + the_map[i, j, 2] + the_map[i, j, 3])/4

        # Combined
        combined_img *=255/max
        combined_img = combined_img.astype(np.uint8)
        img_combined = Image.fromarray(combined_img, 'L')
        img_combined.show()
        '''

        # B1
        the_map[:, :, 0]*=255
        b1 = the_map[:, :, 0]
        min = np.amin(b1)
        b1 = b1 - min
        max = np.amax(b1)
        b1 = b1*255/max
        b1 = b1.astype(np.uint8)

        print("b1 data type: ", b1.dtype)
        img_b1 = Image.fromarray(b1, 'L')
        img_b1.show()

        # Infrared
        # the_map[:, :, 7]*=255
        # infrared = the_map[:, :, 7].astype(np.uint8)
        # print("data type: ", infrared.dtype)
        # print("after max infr: ", np.amax(infrared))
        # print("after min infr: ", np.amin(infrared))
        # img_infr = Image.fromarray(infrared, 'L')
        # img_infr.show()

        # B09
        '''
        the_map[:, :, 9]*=255
        b9 = the_map[:, :, 9]
        min = np.amin(b9)
        b9 = b9 - min
        max = np.amax(b9)
        b9 = b9*255/max
        b9 = b9.astype(np.uint8)

        print("b9 data type: ", b9.dtype)
        img_b9 = Image.fromarray(b9, 'L')
        img_b9.show()
        '''

        '''
        # B10
        the_map[:, :, 10]*=255
        b10 = the_map[:, :, 9]
        min = np.amin(b10)
        b10 = b10 - min
        max = np.amax(b10)
        b10 = b10*255/max
        
        b10 = b10.astype(np.uint8)

        print("b10 data type: ", b10.dtype)
        print("after max b10: ", np.amax(b10))
        print("after min b10: ", np.amin(b10))
        img_b10 = Image.fromarray(b10, 'L')
        img_b10.show()
        '''
        # B11
        # the_map[:, :, 11]*=255
        # b11 = the_map[:, :, 11].astype(np.uint8)
        # print("b11 data type: ", b11.dtype)
        # print("after max b11: ", np.amax(b11))
        # print("after min b11: ", np.amin(b11))
        # img_b11 = Image.fromarray(b11, 'L')
        # img_b11.show()

        # B12
        # the_map[:, :, 12]*=255
        # b12 = the_map[:, :, 12].astype(np.uint8)
        # print("b12 data type: ", b12.dtype)
        # print("after max b12: ", np.amax(b12))
        # print("after min b12: ", np.amin(b12))
        # img_b12 = Image.fromarray(b12, 'L')
        # img_b12.show()



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


    #     """
    #     map[:, :, 4] = gaussian_filter(map[:, :, 4], sigma=3)
    #     for idx, x in np.ndenumerate(map[:, :, 4]):
    #         if map[idx[0], idx[1], 4] > 0: # if non-zero, indicates some presence of clouds
    #             map[idx[0], idx[1], 0] = 0 # R
    #             map[idx[0], idx[1], 1] = 0 # G
    #             map[idx[0], idx[1], 2] = 0 # B
    #             map[idx[0], idx[1], 3] = 0 # IR
    #     for i in range(0, 5):  # R, G, B, IR all need to be normalized
    #         min = np.amin(map[:, :, i]) # i==4 has range between 0-100
    #         map[:, :, i] -= min  # center min = 0
    #         max = np.amax(map[:, :, i])
    #         map[:, :, i] *= (255/max) # converts to range 0, 255
    #     """
    #     # maps[m] = map
    #     # map = map.astype(np.uint8) # convert to int8


    #     # detected_clouds = np.copy(map[:, :, 5])

        # for i in range(0, map.shape[0]):
            # for j in range(0, map.shape[1]):

                # Using SCL, CLP, set land to be black, we will reclaim cloud lost on land from this later on
                # We're doing this here before SCL values are changed later
                # if (map[i, j, 4] == 4 or map[i, j, 4] == 5 or map[i, j, 4] == 0 or map[i, j, 4] == 1 or map[i, j, 4] == 2) and map[i, j, 5]/255 >= 0:
                #     detected_clouds[i, j] = 0
                
                # #SCL layer
                # if map[i, j, 4] == 10: # cirrus clouds
                #     map[i, j, 4] = 225
                # elif map[i, j, 4] == 7: # low prob. clouds
                #     map[i, j, 4] = 235
                # elif map[i, j, 4] == 8: # med prob. clouds
                #     map[i, j, 4] = 245
                # elif map[i, j, 4] == 9:# high prob. clouds
                #     map[i, j, 4] = 255
                # elif map[i, j, 4] == 4 or map[i, j, 4] == 5 or map[i, j, 4] == 0 or map[i, j, 4] == 1 or map[i, j, 4] == 2: # ground/defect pixels
                #     map[i, j, 4] = 100
                # else: 
                #     # water is black, but not completely 0
                #     map[i, j, 4] = 0

                #     # Make our marked clouds array's water also to completely black except for where there are clouds
                #     if detected_clouds[i, j] < 14: # Where 14 is the cut off threshold
                #         detected_clouds[i, j] = 0


                # #CLP
                # if map[i, j, 5]/255 >= 0.25:
                #     map[i, j, 5] = 225
                # else:
                #     map[i, j, 5] = 0


                # clp layer    
                # if map[i, j, 5] < 4: # guarantee not cloud
                #     map[i, j, 5] = 4
                
        # map[:, :, 5] - 4 # min of 0
        # min_clp = 4
        # max_clp = np.amax(map[:, :, 5]) # somewhere around ~251, lets say
        # map[:, :, 5]*(255/max_clp) # gives a ceil of 255

        # find definite clouds over land set to 255 and then blur it

                
        #map[:, :, 5] = map[:, :, 5]*2.55 # clouds = 7, 8,9

        # img_rgb = Image.fromarray(map[:, :, [2, 1, 0]], 'RGB')
        # img_b08 = Image.fromarray(map[:, :, 3], 'L') # 
        # img_scl = Image.fromarray(map[:, :, 4], 'L') # 
        # img_clp = Image.fromarray(map[:, :, 5], 'L')
        # img_aot = Image.fromarray(map[:, :, 6], 'L') # useless
        # img_cld = Image.fromarray(map[:, :, 7], 'L') # useless
        # img_detected_clouds = Image.fromarray(detected_clouds, 'L')

        # img_rgb.show()
        # img_b08.show()
        # img_scl.show()
        # img_clp.show()
        # img_aot.show()
        # img_cld.show()
        # img_detected_clouds.show()


    # maps = maps.astype(np.uint8) # multiple days of maps


    """
    # generate folder
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    for i in range(0, 4):
        col = ''
        if i == 0:
            col = 'B'
        elif i == 1:
            col = 'G'
        elif i == 2:
            col = 'R'
        elif i == 3:
            col = 'IR'

        data_path = folder_path + '/'+col + 
            '.png'  # 106.0_13.0_1.jpg
        print(data_path)
        im = Image.fromarray(bands[:, :, i])
        im.save(data_path)

    # img = Image.fromarray(bands[:, :, [2, 1, 0]], 'RGB')
    # img.show()
    img.save(folder_path+'/col.png')
    """
        #     else:
        #         print("PATH EXISTS ALREADY...")
        #     y = round(y+y_delta, 1)
        #     break
        # break
        # x = round(x+x_delta, 1)
        # y = start_y
# plt.show()