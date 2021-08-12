import numpy as np
import rasterio
import os

filelist = []


path = "~/wet_garbage/maps"

for root, dirs, files in os.walk(path):
    for file in files:
        filelist.append(os.path.join(root,file))
        
for name in filelist:
    print(name)

