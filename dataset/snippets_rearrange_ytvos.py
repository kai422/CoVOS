#DATASET_ROOT=. python encode_ytvos.py
import os
import subprocess
from path_config import config

root_folder = os.path.join(config.data_path(),'YouTube-VOS')
images_path_train = os.path.join(root_folder, 'train_all_frames','JPEGImages')
rearrange_path_train = os.path.join(root_folder, 'train_all_frames','JPEGImages_rearrange')
if not os.path.exists(rearrange_path_train):
	os.makedirs(rearrange_path_train)

images_path_valid = os.path.join(root_folder, 'valid_all_frames','JPEGImages')
rearrange_path_valid = os.path.join(root_folder, 'valid_all_frames','JPEGImages_rearrange')
if not os.path.exists(rearrange_path_valid):
	os.makedirs(rearrange_path_valid)


for video_id in sorted(os.listdir(images_path_train)):
    png_list = sorted(os.listdir(os.path.join(images_path_train, video_id)))
    base_frame_index = None
    snippets_frame_id = 0
    for i, name in enumerate(png_list):
        if '.jpg' not in name:
            continue
        frame_id = int(name.split('.')[0])
        if base_frame_index is None:
            snippets_frame_id = 0
            base_frame_index = frame_id
        elif base_frame_index + snippets_frame_id + 1 == frame_id:
            snippets_frame_id += 1
        else:
            snippets_frame_id = 0
            base_frame_index = frame_id
        out_image_name = "{:05d}_{:05d}.jpg".format(base_frame_index, snippets_frame_id)
        src = os.path.join(os.path.join(images_path_train, video_id), name)
        dst = os.path.join(os.path.join(rearrange_path_train, video_id), out_image_name)
        command = "cp {} {}".format(src, dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(command)
        subprocess.run(command, shell=True, check=True)
        assert(os.path.isfile(dst))

for video_id in sorted(os.listdir(images_path_valid)):
    png_list = sorted(os.listdir(os.path.join(images_path_valid, video_id)))
    base_frame_index = None
    snippets_frame_id = 0
    for i, name in enumerate(png_list):
        frame_id = int(name.split('.')[0])
        if base_frame_index is None:
            snippets_frame_id = 0
            base_frame_index = frame_id
        elif base_frame_index + snippets_frame_id + 1 == frame_id:
            snippets_frame_id += 1
        else:
            snippets_frame_id = 0
            base_frame_index = frame_id
        out_image_name = "{:05d}_{:05d}.jpg".format(base_frame_index, snippets_frame_id)
        src = os.path.join(os.path.join(images_path_valid, video_id), name)
        dst = os.path.join(os.path.join(rearrange_path_valid, video_id), out_image_name)
        command = "cp {} {}".format(src, dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(command)
        subprocess.run(command, shell=True, check=True)
        assert(os.path.isfile(dst))