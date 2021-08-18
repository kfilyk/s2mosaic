import subprocess
import os
import shlex
path = "./maps/"
files = []
filelist = []
for dirpath, dirnames, filenames in os.walk("./maps"):
  for filename in [f for f in filenames if f.endswith(".png")]:
    filelist.append(os.path.join(dirpath,filename))
    files.append(filename)
print(files)
print(filelist)
for x in filelist:
  project_list_command = ("gcloud compute scp %s raghavkhurana@instance-1:~ --zone us-central1-a" % x)
  project_output = subprocess.check_output(project_list_command,shell = True)
  print(project_output)