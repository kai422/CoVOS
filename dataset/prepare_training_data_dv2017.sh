#!/bin/bash
PYTHONPATH=$PWD python dataset/prepare_training_data.py --dset dv2017 --dset_path \
  /mnt/ssd/allusers/kai422/dataset_VOS/DAVIS/2017/trainval  \
  --save_path /mnt/ssd/allusers/kai422/dataset_VOS/DAVIS/2017/trainval/HEVCfeatures/train/dv2017