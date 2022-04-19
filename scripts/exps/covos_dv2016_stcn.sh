#!/bin/bash
PYTHONPATH=$PWD python covos.py --base_segmentor STCN_DV16 --dset dv2016 --save_path results/covos_stcn \
--cfg model=model_zoo/STCN/saves/stcn.pth top=20 mem_every=2
