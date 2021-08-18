
import os
import socket
import tqdm

SEPARATOR = "<SEPARATOR>"

BUFFER_SIZE = 4096


host = "0.0.0.0"

port = 4000

s = socket.socket()
s.bind((host,port))
print(f"[+] Connecting to {host}:{port}")
s.listen(5)
print(f"[*] Listening as{host}:{port}")
client_socket, address = s.accept()
print(f"[+] {address} is connected")

while 1:
    received = client_socket.recv(BUFFER_SIZE).decode()
    print(received)
    filename, filesize = received.split(SEPARATOR)
    filename = os.path.basename(filename)
    # convert to integer
    filesize = int(filesize)
    with open(filename, "wb") as f:
        bytes_read =  client_socket.recv(BUFFER_SIZE)
        if not bytes_read:    
            # nothing is received
            # file transmitting is done
            break
        f.write(bytes_read)


client_socket.close()
s.close()
