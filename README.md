# [CoVOS](https://arxiv.org/abs/2107.12192)

Offical PyTorch implementation of **CoVOS**:

[Accelerating Video Object Segmentation with Compressed Video](https://arxiv.org/abs/2107.12192), CVPR'22.    
[Kai Xu](#), [Angela Yao](https://www.comp.nus.edu.sg/~ayao/)    
Computer Vision and Machine Learning group, NUS.   
\[[PDF](https://arxiv.org/abs/2107.12192)\] \[[Project Page](https://kai422.github.io/CoVOS/)\]



## Installation
***Prepare Conda Environment**: (We test the code for python=3.10 and pytorch=1.11. Similar versions will also work.)* 

```bash
conda create -n CoVOS python
conda activate CoVOS
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install tqdm tabulate opencv-python easydict ninja scikit-image scikit-video
# Install CUDA motion vector warping function.
python setup.py build_ext --inplace install
```

***Prepare HEVC feature decoder:** (Here are two options.)*

* Compile from source code and update `{install_path}/usr/local/bin/hevc` to `hevc_feature_decoder_path` in [path_config.py](path_config.py) .
```bash
git clone https://github.com/kai422/openHEVC_feature_decoder.git
cd openHEVC_feature_decoder
git checkout Interface_MV_Residual
# If yasm package is not installed, use the following command. 
sudo apt-get install -y yasm
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make -j9
make DESTDIR={install_path} install
```
* Use pre-compiled binary files for ubuntu 18.04 at [decoder/bin/hevc](decoder/bin). You don't need to update the path.



## Prepare Data:

***Download Data:***

*DAVIS:* Download [480p](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and [Full-Resolution](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip) data and put them into the same folder.
After unzipping, the structure of the directory should be:
```
{data_path}/
├──DAVIS/
│   ├──Annotations
│   │   └── ... 
│   ├──ImageSets
│   │   └── ...  
│   └──JPEGImages
│       ├──480p
│       └──Full-Resolution
```


*YouTube-VOS:* Download [YouTubeVOS 2018](https://youtube-vos.org/dataset/). After unzipping, the structure of the directory should be:
```
{data_path}/
├──YouTube-VOS/
│   ├──train/
│   ├──train_all_frames/
│   ├──valid/
│   └──valid_all_frames/
```
Some video frame indexes do not start from 0, so we need to rearrange the snippets.
```
bash scripts/snippets_rearrange.sh
```

Update `data_path` in [path_config.py](path_config.py).  



***Encode Videos:***

Encode raw image sequences into HEVC videos by
```bash
# to reproduce, use FFmpeg 3.4.8-0ubuntu0.2 (the default version for ubuntu 18.04)
bash scripts/data/encode_video_davis.sh
bash scripts/encode_video_ytvos.sh
```
Encoded videos will be stored at `{data_path}/DAVIS/HEVCVideos` and `{data_path}/YouTube-VOS/HEVCVideos`.



## Running
```bash
# DAVIS16, base model: stm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_stm.sh
RESULT_PATH=results/covos_stm/dv2016 DSET=dv2016val python evaluate_from_folder.py

# DAVIS16, base model: frtm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_frtm.sh
RESULT_PATH=results/covos_frtm/dv2016 DSET=dv2016val python evaluate_from_folder.py

# DAVIS16, base model: mivos
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_mivos.sh
RESULT_PATH=results/covos_mivos/dv2016 DSET=dv2016val python evaluate_from_folder.py

# DAVIS16, base model: stcn
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_stcn.sh
RESULT_PATH=results/covos_stcn/dv2016 DSET=dv2016val python evaluate_from_folder.py 


# DAVIS17, base model: stm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2017_stm.sh
RESULT_PATH=results/covos_stm/dv2017 DSET=dv2017val python evaluate_from_folder.py

# DAVIS17, base model: frtm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2017_frtm.sh
RESULT_PATH=results/covos_frtm/dv2017 DSET=dv2017val python evaluate_from_folder.py

# DAVIS17, base model: mivos
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2017_mivos.sh
RESULT_PATH=results/covos_mivos/dv2017 DSET=dv2017val python evaluate_from_folder.py

# DAVIS17, base model: stcn
scripts/exps/covos_dv2017_stcn.sh
RESULT_PATH=results/covos_stcn/dv2017 DSET=dv2017val python evaluate_from_folder.py


#Youtube-VOS 2018, base model: frtm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_yt2018_frtm.sh

# Youtube-VOS 2018, base model: mivos
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_yt2018_mivos.sh

# Youtube-VOS 2018, base model: stcn
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_yt2018_stcn.sh
```

## Training



## License and Acknowledgement
This project is released under the GPL-3.0 License.  We refer to codes from [`MiVOS`](https://github.com/hkchengrex/MiVOS), [`FRTM-VOS`](https://github.com/andr345/frtm-vos), and [`DAVIS`](https://github.com/davisvideochallenge/davis2017-evaluation).


## Citation


```bibtex
@inproceedings{xu2022covos,
  title={Accelerating Video Object Segmentation with Compressed Video},
  author={Kai Xu and Angela Yao},
  booktitle={CVPR},
  year={2022}
}
```