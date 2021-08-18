import numpy as np
import socket
import tqdm
import rasterio
import os
import zipfile
import cv2
import subprocess

filelist = []
jp_data = []

SEPARATOR = " "
BUFFER_SIZE = 2048
SERVER_PORT = 4000
SERVER_HOST = "34.133.186.193"
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'

s = socket.socket()
files = []
s = socket.socket()
for dirpath, dirnames, filenames in os.walk("./maps"):
    for filename in [f for f in filenames if f.endswith(".png")]:
        filelist.append(os.path.join(dirpath,filename))
        files.append(filename)

print(files)
print(filelist)
for f in filelist:
    subprocess.run(["scp", f, "USER@SERVER:PATH"])

'''
s.connect((SERVER_HOST, SERVER_PORT))
print("[+] Connected.")
for ind,img in enumerate(filelist):
    filename = files[ind]
    filesize = os.path.getsize(img)
    file = open(img, 'rb')
    print(filename + ' ' + filesize)

    s.sendall(f"{filename}{SEPARATOR}{filesize}".encode())
    
    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    while True:
        # read the bytes from the file
        bytes_read = file.read(BUFFER_SIZE)
        if not bytes_read:
            # file transmitting is done
            break
        # we use sendall to assure transimission in 
        # busy networks
        s.sendall(bytes_read)
        # update the progress bar
        progress.update(len(bytes_read))

s.close()
'''