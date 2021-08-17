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
SERVER_PORT = 5001
SERVER_HOST = "192.168.171.36"

s = socket.socket()
s.bind((SERVER_HOST,SERVER_PORT))
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
s.listen(5)
while 1:
	print(f"[*] Listening as{SERVER_HOST}:{SERVER_PORT}")
	client_socket, address = s.accept()
	print(f"[+] {address} is connected")
	

	

