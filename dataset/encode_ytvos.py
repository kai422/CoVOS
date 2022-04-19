import os
import subprocess

from path_config import config

root_folder = os.path.join(config.data_path(),'YouTube-VOS')
images_path_train = os.path.join(root_folder, 'train_all_frames','JPEGImages_rearrange')
videos_path_train = os.path.join(root_folder, 'train_all_frames','HEVCVideos')
if not os.path.exists(videos_path_train):
	os.makedirs(videos_path_train)

images_path_valid = os.path.join(root_folder, 'valid_all_frames','JPEGImages_rearrange')
videos_path_valid = os.path.join(root_folder, 'valid_all_frames','HEVCVideos')
if not os.path.exists(videos_path_valid):
	os.makedirs(videos_path_valid)

for i in sorted(os.listdir(images_path_train)):
	png_list = sorted(os.listdir(os.path.join(images_path_train, i)))
	sequence_id = png_list[0].split('_')[0]
	try:
		command = "/usr/bin/ffmpeg -framerate 30 -i {}/{}_%05d.jpg -c:v libx265 -pix_fmt yuv420p {}/{}_{}.mp4".format(os.path.join(images_path_train,i), sequence_id, videos_path_train, i, sequence_id)
		print(command)
		subprocess.run(command, shell=True, check=True)
	except subprocess.CalledProcessError:
		exit()

for i in sorted(os.listdir(images_path_valid)):
	png_list = sorted(os.listdir(os.path.join(images_path_valid, i)))
	sequence_id = png_list[0].split('_')[0]
	try:
		command = "/usr/bin/ffmpeg -framerate 30 -i {}/{}_%05d.jpg -c:v libx265 -pix_fmt yuv420p {}/{}_{}.mp4".format(os.path.join(images_path_valid,i), sequence_id, videos_path_valid, i, sequence_id)
		print(command)
		subprocess.run(command, shell=True, check=True)
	except subprocess.CalledProcessError:
		exit()

