import numpy as np
import rasterio
import os

filelist = []


path = "/home/ubuntu/wet_garbage/maps/"

for files in os.listdir(path):
	for data in os.listdir(path+files):
		if(data.endswith(".jp2")):
   			 filelist.append(data)
        
for name in filelist:
	with rasterio.open(jp2) as f:
        jp_data.append(f.read(1))
        np_data = np.array(jp_data, dtype=jp_data[0].dtype)
        print(np_data)
