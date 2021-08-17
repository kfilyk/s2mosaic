import numpy as np
import socket
import tqdm
import rasterio
import os
import zipfile

filelist = []
jp_data = []

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096
SERVER_PORT = 4000
SERVER_HOST = "34.133.186.193"

s = socket.socket()
path = "/home/ubuntu/wet_garbage/maps/"

for files in os.listdir(path):
	for data in os.listdir(path+files):
		if(data.endswith(".jp2")):
   			 filelist.append(data)
print(filelist)


for name in filelist:
	with rasterio.open(jp2) as f:
         jp_data.append(f.read(1))
         np_data = np.array(jp_data, dtype=jp_data[0].dtype)
         print(np_data)


s.connect((host, port))
print("[+] Connected.")


	

