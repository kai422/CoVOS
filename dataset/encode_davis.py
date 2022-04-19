import os
from path_config import path_config

image_path = os.path.join(path_config.data_path(),'DAVIS/JPEGImages/Full-Resolution')
video_save_path = os.path.join(path_config.data_path(),'DAVIS/HEVCVideos')
os.makedirs(video_save_path, exist_ok=True)

for sequence in sorted(os.listdir(image_path)):
    os.system("/usr/bin/ffmpeg -framerate 24 -i {}/{}/%05d.jpg -c:v libx265 -pix_fmt yuv420p {}/{}.mp4".format(image_path, sequence,video_save_path, sequence))