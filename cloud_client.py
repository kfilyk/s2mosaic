import os
import socket
import tqdm

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096

host = "192.168.171.36"

port = 5001

s = socket.socket()
print(f"[+] Connecting to {host}:{port}")

s.connect((host, port))
print("[+] Connected.")
