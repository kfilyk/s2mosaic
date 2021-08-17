# import rasterio
# import geopyspark as gps
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import dateutil.relativedelta
# from pyspark import SparkContext
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, WmsRequest, WcsRequest, MimeType, CRS, BBox, DataCollection, MimeType, SentinelHubRequest, SentinelHubDownloadClient, bbox_to_dimensions, DownloadRequest
from PIL import Image
from scipy.ndimage.filters import gaussian_filter


# b02 red b03 green b04 blue ; CLD = sentinel2 cloud detection
evalscript_10m_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "SCL", "CLD"],
            }],
            output: {
                bands: 5,
                sampleType: "UINT8",
            },
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02*255, sample.B03*255, sample.B04*255, sample.B08*255, sample.SCL, sample.CLD];

    }

"""
#mosaicking: "ORBIT"


def get_map_request(time_interval):
    return SentinelHubRequest(
        evalscript=evalscript_10m_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,  # L2A atmospheric corrected data
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
sites['accra'] = [-0.40, 5.20, 0.20, 5.60]
sites['accra'] = [-0.20, 5.40, 0.20, 5.60]

for s in sites:
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

    while (x < end_x):
        while (y < end_y):
            folder_path = './maps/'+s + '/' + str(x) + '_'+str(y)
            if not os.path.exists(folder_path):
                coords = [x, y, x+x_delta, y+y_delta]
                # betsiboka_coords_wgs84 = [46.15, -16.20, 46.35, -16.00]  # 0.2 x0.2 box

                betsiboka_bbox = BBox(bbox=coords, crs=CRS.WGS84)
                betsiboka_size = bbox_to_dimensions(
                    betsiboka_bbox, resolution=resolution)

                """All requests require bounding box to be given as an instance of sentinelhub.geometry.BBox with corresponding Coordinate Reference System (sentinelhub.geometry.CRS)"""
                # get one month of data
                today = datetime.today()
                interval_length = 7 # 7 days
                earliest_date = today + dateutil.relativedelta.relativedelta(weeks=-1)
                last_week = today + dateutil.relativedelta.relativedelta(days=-interval_length) # one week before today
                slots = [] # all weeks of data to be queried
                while today > earliest_date:
                    today_str = today.strftime('%Y-%m-%d')  # get in string format
                    last_week_str = last_week.strftime('%Y-%m-%d')
                    slots.append((last_week_str, today_str)) # 1 week intervals
                    today = last_week
                    last_week = today + dateutil.relativedelta.relativedelta(days=-interval_length)
                print(slots) # should have a number of time intervals

                list_of_requests = [get_map_request(slot) for slot in slots]
                list_of_requests = [request.download_list[0] for request in list_of_requests] # now have a list of requests
                # download data with multiple threads
                raw_maps = np.array(SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5), dtype=np.float64) # ACTUALLY DOWNLOADS
                #raw_maps = np.array(map_requests.get_data(), dtype=np.float64) # get array of all maps - note that SCL cloud cover map
                print(raw_maps.shape)  # should be like (3, 13259,1399, 5)
                print(raw_maps.dtype)
                maps = np.copy(raw_maps)
                average_map = np.zeros_like(maps[0, :, :, :4]) # create a single map instance of size (w, h, 4) instead of 5
                pixel_count = np.zeros_like(maps[0, :, :]) # counts how many different sub map values have been accumulated for a given pixel location
                print(average_map.shape)  # should be like (3, 13259,1399, 5)
                print(pixel_count.shape)  # should be like (3, 13259,1399, 5)


                
                for m in range(0, len(raw_maps)):
                    map = raw_maps[m] 
                    """
                    map[:, :, 4] = gaussian_filter(map[:, :, 4], sigma=3)
                    for idx, x in np.ndenumerate(map[:, :, 4]):
                        if map[idx[0], idx[1], 4] > 0: # if non-zero, indicates some presence of clouds
                            map[idx[0], idx[1], 0] = 0 # R
                            map[idx[0], idx[1], 1] = 0 # G
                            map[idx[0], idx[1], 2] = 0 # B
                            map[idx[0], idx[1], 3] = 0 # IR
                    for i in range(0, 5):  # R, G, B, IR all need to be normalized
                        min = np.amin(map[:, :, i]) # i==4 has range between 0-100
                        map[:, :, i] -= min  # center min = 0
                        max = np.amax(map[:, :, i])
                        map[:, :, i] *= (255/max) # converts to range 0, 255
                    """
                    maps[m] = map
                    map = map.astype(np.uint8) # convert to int8
                    img_rgb = Image.fromarray(map[:, :, [2, 1, 0]], 'RGB')

                    for i in range(0, map.shape[0]):
                        for j in range(0, map.shape[1]):
                            if map[i, j, 4] == 4 or map[i, j, 4] == 5 or map[i, j, 4] == 0 or map[i, j, 4] == 1 or map[i, j, 4] == 2:
                                map[i, j, 4] = 255
                            
                    #map[:, :, 4] = map[:, :, 4]*255 # clouds = 7, 8,9

                    #map[:, :, 4] = map[:, :, 4]*2.55 # clouds = 7, 8,9
                    img_cld = Image.fromarray(map[:, :, 4], 'L') # show clouds

                    img_rgb.show()
                    img_cld.show()
                    #img_ir = Image.fromarray(map[:, :, 3], 'L')
                    #img_ir.show()
                    #img_scl = Image.fromarray(map[:, :, 4], 'L')
                    #img_scl.show()
                maps = maps.astype(np.uint8) # multiple days of maps


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
            else:
                print("PATH EXISTS ALREADY...")
            y = round(y+y_delta, 1)
            break
        break
        x = round(x+x_delta, 1)
        y = start_y
