import os
import zipfile

for file in os.listdir():
    print(file)
    if(file.endswith(".zip")):
        with zipfile.ZipFile(file, 'r') as zip_ref:
	    zip_ref.extractall('.')
        os.remove(file)

