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

