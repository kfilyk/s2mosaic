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
	print(name)

