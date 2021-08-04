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

evalscript_10m_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "CLM"],
                units: "DN"
            }],
            output: {
                bands: 5,
                sampleType: "INT16"
            },
            mosaicking: "ORBIT"
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08, sample.CLM];
    }
"""

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
sites['hanoi'] = [106.00, 13.00, 116.00, 22.50]

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
                today = datetime.today()
                last_month = (
                    today + dateutil.relativedelta.relativedelta(months=-36))

                today = today.strftime('%Y-%m-%d')  # get in string format
                last_month = last_month.strftime('%Y-%m-%d')
                print(today)
                print(last_month)

                request_bands = SentinelHubRequest(
                    evalscript=evalscript_10m_bands,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L2A,  # L2A atmospheric corrected data
                            time_interval=(last_month, today),
                            # mosaicking_order='leastCC'
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

                # int16 type needs to be scaled by 1/10000. then standardize betweeon 0, 256
                bands = np.array(request_bands.get_data()[0]) / 10000
                for i in range(0, 5):
                    min = np.amin(bands[:, :, i])
                    max = np.amax(bands[:, :, i])
                    print("MIN/MAX OG: ")
                    print(min)
                    print(max)
                    bands[:, :, i] = bands[:, :, i]-min  # center at zero
                    max = np.amax(bands[:, :, i])
                    bands[:, :, i] *= (255/max)
                    print(np.amax(bands[:, :, i]))
                    print(np.amin(bands[:, :, i]))
                bands = bands.astype(np.uint8)

                # generate folder
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                for i in range(0, 5):
                    col = ''
                    if i == 0:
                        col = 'B'
                    elif i == 1:
                        col = 'G'
                    elif i == 2:
                        col = 'R'
                    elif i == 3:
                        col = 'IR'
                    else:
                        col = 'CLM'

                    data_path = folder_path + '/'+col + \
                        '.png'  # 106.0_13.0_1.jpg
                    print(data_path)
                    im = Image.fromarray(bands[:, :, i])
                    im.save(data_path)

                img = Image.fromarray(bands[:, :, [2, 1, 0]], 'RGB')
                # img.show()
                img.save(folder_path+'/col.png')
            else:
                print("PATH EXISTS ALREADY...")
            y = round(y+y_delta, 1)
        x = round(x+x_delta, 1)
        y = start_y
