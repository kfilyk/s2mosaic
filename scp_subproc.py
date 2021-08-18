import subprocess
import os
import shlex


path = "./maps/"
files = []
filelist = []

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".png")]:
        filelist.append(os.path.join(dirpath,filename))
        files.append(filename)

print(files)
print(filelist)

for x in filelist:
    project_list_command = ("scp -i ~/home/ubuntu/.ssh %s raghavkhurana@34.133.186.193:/~ " % x)
    project_output = subprocess.check_output(shlex.split(project_list_command))
    print(project_output)
