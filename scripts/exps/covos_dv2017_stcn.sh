#!/bin/bash
PYTHONPATH=$PWD  python covos.py --base_segmentor STCN_DV17 --dset dv2017 --save_path results/covos_stcn \
--cfg model=model_zoo/STCN/saves/stcn.pth top=20 mem_every=2
