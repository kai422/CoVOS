#!/bin/bash
PYTHONPATH=$PWD python covos.py --base_segmentor MIVOS_DV16 --dset dv2016  --save_path results/covos_mivos \
--cfg model=model_zoo/MiVOS/saves/topk_stm.pth top_k=50 mem_every=2 use_km=0
