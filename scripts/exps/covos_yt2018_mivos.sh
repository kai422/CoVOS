#!/bin/bash
PYTHONPATH=$PWD python covos_yt.py --base_segmentor MIVOS_YTVOS --dset yt2018  --save_path results/covos_mivos \
--cfg model=model_zoo/MiVOS/saves/topk_stm.pth top_k=50 mem_every=5 use_km=0
