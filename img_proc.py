import numpy as np
import socket
import tqdm
import rasterio
import os
import zipfile
import cv2

filelist = []
jp_data = []

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096
SERVER_PORT = 4000
SERVER_HOST = "34.133.186.193"
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'

path = "/home/ubuntu/wet_garbage/maps/"
files = []
s = socket.socket()
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".jp2")]:
        filelist.append(os.path.join(dirpath,filename))
        if "10m" in filename:
            files.append(filename)


NTL = []

for name in filelist:
    if "NTL" in name:
        if "10m" in name:
            NTL.append(name)
print(files)
print(NTL)

png = []
for x,ele in enumerate(NTL):
    temp_im = cv2.imread(ele)
    cv2.imwrite("%s.png"%ele[:-4],temp_im)
    png.append("%s.png"%ele[:-4])

s.connect((SERVER_HOST, SERVER_PORT))
print("[+] Connected.")
print(png)
for ind,img in enumerate(png):
    filename = img
    filesize = os.path.getsize(filename)
    print(filename)
    s.sendall(f"{filename}{SEPARATOR}{filesize}".encode("utf-8"))
    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in 
            # busy networks
            s.sendall(bytes_read)
            # update the progress bar
        progress.update(len(bytes_read))

s.close()

