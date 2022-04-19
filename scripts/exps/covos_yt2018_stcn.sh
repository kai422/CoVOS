#!/bin/bash
PYTHONPATH=$PWD python covos_yt.py --base_segmentor STCN_YTVOS --dset yt2018  --save_path results/covos_stcn \
--cfg model=model_zoo/STCN/saves/stcn.pth top=20 mem_every=5
