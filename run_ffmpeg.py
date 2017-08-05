import os
import subprocess
import sys

folder = "./halfcheetah"

for sub_folder in os.listdir(folder):
    path = os.path.join(folder, sub_folder)
    cmd1 = "ffmpeg -start_number 0 -i {}/ob_raw_%d.jpg -vcodec mpeg4 {}/{}_{}_raw.avi".format(path, folder, folder, sub_folder)
    cmd2 = "ffmpeg -start_number 0 -i {}/ob_%d.jpg -vcodec mpeg4 {}/{}_{}.avi".format(path, folder, folder, sub_folder)
    subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)
    subprocess.Popen(cmd2.split(), stdout=subprocess.PIPE)
