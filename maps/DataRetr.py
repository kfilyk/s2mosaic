#import rasterio
#import geopyspark as gps
import numpy as np
import datetime

#from pyspark import SparkContext
import matplotlib.pyplot as plt
from sentinelhub import SHConfig
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, DataCollection

config = SHConfig()

if config.instance_id == '':
    print("Warning! To use OGC functionality of Sentinel Hub, please configure the `instance_id`.")


def plot_image(image, factor=1):
    """
    Utility function for plotting RGB images.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image * factor, 1))
    else:
        plt.imshow(image)


sample_coords = [46.16, -16.15, 46.51, -15.580]

"""All requests require bounding box to be given as an instance of sentinelhub.geometry.BBox with corresponding Coordinate Reference System (sentinelhub.geometry.CRS)"""

sample_box = BBox(bbox=sample_coords, crs=CRS.WGS84)


wms_true_color_request = WmsRequest(
    data_collection=DataCollection.SENTINEL2_L1C,
    data_folder='.',
    layer='1_TRUE_COLOR',
    bbox=sample_box,
    time='2017-12-15',
    width=512,
    height=856,
    config=config
)

wms_true_color_request.save_data()

#wcs_true_color_img = wcs_true_color_request.get_data()
wms_true_color_img = wms_true_color_request.get_data()
print('Single element in the list is of type = {} and has shape {}'.format(
    type(wms_true_color_img[-1]), wms_true_color_img[-1].shape))

plot_image(wms_true_color_img[-1])
