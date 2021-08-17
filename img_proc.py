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


for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".jp2")]:
        filelist.append(os.path.join(dirpath,filename))

NTL = [:3]

for name in filelist:
    if "NTL" in name:
        if "10m" in name:
            NTL[0].append(name)
        if "20m" in name:
            NTL[1].append(name)
        if "60m" in name:
            NTL[2].append(name)


data = np.array(jp_data, dtype=jp_data[0].dtype)
print(data.shape)
s.connect((SERVER_HOST, SERVER_PORT))
print("[+] Connected.")
filename = "np_array"
filesize = os.path.getsize(filename)
s.send(f"{filename}{SEPARATOR}{filesize}".encode())

