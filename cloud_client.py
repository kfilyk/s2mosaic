
import os
import socket
import tqdm

SEPARATOR = " "

BUFFER_SIZE = 2048


host = "0.0.0.0"

port = 4000

s = socket.socket()
s.bind((host,port))
print(f"[+] Connecting to {host}:{port}")
s.listen(1)
print(f"[*] Listening as{host}:{port}")
client_socket, address = s.accept()
print(f"[+] {address} is connected")

while 1:
    received = client_socket.recv(BUFFER_SIZE)
    print(received)
    filename, filesize = received.decode().split(SEPARATOR)
    filename = os.path.basename(filename)
    # convert to integer
    filesize = int(filesize)
    print(filename)
    with open(filename, "wb") as f:
        while 1:

            bytes_read =  client_socket.recv(BUFFER_SIZE)
            if not bytes_read:
                break
        
            # nothing is received
           # file transmitting is done
            f.write(bytes_read)
        f.close()

client_socket.close()
s.close()
