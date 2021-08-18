import numpy as np
import socket
import tqdm
import rasterio
import os
import zipfile
import cv2

filelist = []
jp_data = []

SEPARATOR = " "
BUFFER_SIZE = 4096
SERVER_PORT = 4000
SERVER_HOST = "34.133.186.193"
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'

s = socket.socket()
path = "./maps/"
files = []
s = socket.socket()
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".png")]:
        filelist.append(os.path.join(dirpath,filename))
        files.append(filename)

print(files)
print(filelist)

s.connect((SERVER_HOST, SERVER_PORT))
print("[+] Connected.")
for ind,img in enumerate(filelist):
    filename = img
    filesize = os.path.getsize(filename)
    print(filename)
    s.sendall(f"{filename}{SEPARATOR}{filesize}".encode())
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
