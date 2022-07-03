# Accelerating Video Object Segmentation with Compressed Video

This is an offical PyTorch implementation of 
>**Accelerating Video Object Segmentation with Compressed Video. CVPR 2022.**  
\[[arXiv](https://arxiv.org/abs/2107.12192)\] \[[Project Page](https://kai422.github.io/CoVOS/)\]  
[Kai Xu](https://kai422.github.io/), [Angela Yao](https://www.comp.nus.edu.sg/~ayao/)    
Computer Vision and Machine Learning group, NUS.   




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

* Compile from source code ([openHEVC_feature_decoder](https://github.com/kai422/openHEVC_feature_decoder)) and update `{install_path}/usr/local/bin/hevc` to `hevc_feature_decoder_path` in [path_config.py](path_config.py) .
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

Alternatively, HEVC-encoded video could be downloaded from [Google Drive](https://drive.google.com/drive/folders/1xM6QZbAzcS1LpDQ1uhuJS4Dx7iivJPPK?usp=sharing).

## Models
Download pretrained models for base network:
- FRTM-VOS:
*sh model_zoo/FRTM/weights/download_weights.sh*
- STM:
Download it from [STM repository](https://github.com/seoungwugoh/STM)
and put it at *model_zoo/STM/STM_weights.pth*.
- MiVOS:
Download *s012* model from [MiVOS repository](https://github.com/hkchengrex/Mask-Propagation) and put it at *model_zoo/MiVOS/saves/topk_stm.pth*.
Download *s012* model from [STCN repository](https://github.com/hkchengrex/STCN) and put it at *model_zoo/STCN/saves/stcn.pth*.

CoVOS pretrained models are already included in the uploaded github repository: *weights/covos_light_encoder.pth* and *weights/covos_propagator.pth*.

## Testing
You can download pre-computed results from [Google Drive](https://drive.google.com/drive/folders/1Rqt4KIH510UI3iOS_fxY3IWSMnnuQb2s?usp=sharing).    

**Commands:**

| DAVIS 16 Val | J    | F    | J&F  | FPS  |
|--------------|------|------|------|------|
| STM          | 88.7 | 89.9 | 89.3 | 14.9 |
| STM+CoVOS    | 87.0 | 87.3 | 87.2 | 31.5 |
```bash
# DAVIS16, base model: stm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_stm.sh
RESULT_PATH=results/covos_stm/dv2016 DSET=dv2016val python evaluate_from_folder.py
```
| DAVIS 16 Val   | J    | F    | J&F  | FPS  |
|----------------|------|------|------|------|
| FRTM-VOS       | -    | -    | 83.5 | 21.9 |
| FRTM-VOS+CoVOS | 82.3 | 82.2 | 82.3 | 28.6 |

```bash
# DAVIS16, base model: frtm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_frtm.sh
RESULT_PATH=results/covos_frtm/dv2016 DSET=dv2016val python evaluate_from_folder.py
```

| DAVIS 16 Val | J    | F    | J&F  | FPS  |
|--------------|------|------|------|------|
| MiVOS        | 89.7 | 92.4 | 91.0 | 16.9 |
| MiVOS+CoVOS  | 89.0 | 89.8 | 89.4 | 36.8 |

```bash
# DAVIS16, base model: mivos
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_mivos.sh
RESULT_PATH=results/covos_mivos/dv2016 DSET=dv2016val python evaluate_from_folder.py
```
| DAVIS 16 Val | J    | F    | J&F  | FPS  |
|--------------|------|------|------|------|
| STCN         | 90.4 | 93.0 | 91.7 | 26.9 |
| STCN+CoVOS   | 88.5 | 89.6 | 89.1 | 42.7 |
```bash
# DAVIS16, base model: stcn
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2016_stcn.sh
RESULT_PATH=results/covos_stcn/dv2016 DSET=dv2016val python evaluate_from_folder.py 
```
---

| DAVIS 17 Val | J    | F    | J&F  | FPS  |
|--------------|------|------|------|------|
| STM          | 79.2 | 84.3 | 81.8 | 10.6 |
| STM+CoVOS    | 78.3 | 82.7 | 80.5 | 23.8 |

```bash
# DAVIS17, base model: stm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2017_stm.sh
RESULT_PATH=results/covos_stm/dv2017 DSET=dv2017val python evaluate_from_folder.py
```
| DAVIS 17 Val   | J    | F    | J&F  | FPS  |
|----------------|------|------|------|------|
| FRTM-VOS       | -    | -    | 76.7 | 14.1 |
| FRTM-VOS+CoVOS | 69.7 | 75.2 | 72.5 | 20.6 |

```bash
# DAVIS17, base model: frtm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2017_frtm.sh
RESULT_PATH=results/covos_frtm/dv2017 DSET=dv2017val python evaluate_from_folder.py
```

| DAVIS 17 Val | J    | F    | J&F  | FPS  |
|--------------|------|------|------|------|
| MiVOS        | 81.8 | 87.4 | 84.5 | 11.2 |
| MiVOS+CoVOS  | 79.7 | 84.6 | 82.2 | 25.5 |


```bash
# DAVIS17, base model: mivos
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_dv2017_mivos.sh
RESULT_PATH=results/covos_mivos/dv2017 DSET=dv2017val python evaluate_from_folder.py
```

| DAVIS 17 Val | J    | F    | J&F  | FPS  |
|--------------|------|------|------|------|
| STCN         | 82.0 | 88.6 | 85.3 | 20.2 |
| STCN+CoVOS   | 79.7 | 85.1 | 82.4 | 33.7 |

```bash
# DAVIS17, base model: stcn
scripts/exps/covos_dv2017_stcn.sh
RESULT_PATH=results/covos_stcn/dv2017 DSET=dv2017val python evaluate_from_folder.py

```
| YT-VOS 18 Val  | G    | J_s  | F_s  | J_u  | F_u  | FPS  |
|----------------|------|------|------|------|------|------|
| FRTM-VOS       | 72.1 | 72.3 | 76.2 | 65.9 | 74.1 | 7.7  |
| FRTM-VOS+CoVOS | 65.6 | 68.0 | 71.0 | 58.2 | 65.4 | 25.3 |
```bash
#Youtube-VOS 2018, base model: frtm
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_yt2018_frtm.sh
```
| YT-VOS 18 Val | G    | J_s  | F_s  | J_u  | F_u  | FPS  |
|---------------|------|------|------|------|------|------|
| MiVOS         | 82.6 | 81.1 | 85.6 | 77.7 | 86.2 | 13   |
| MiVOS+CoVOS   | 79.3 | 78.9 | 83.0 | 73.5 | 81.7 | 45.9 |
```bash
# Youtube-VOS 2018, base model: mivos
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_yt2018_mivos.sh
```
| YT-VOS 18 Val | G    | J_s  | F_s  | J_u  | F_u  | FPS  |
|---------------|------|------|------|------|------|------|
| STCN          | 84.3 | 83.2 | 87.9 | 79.0 | 87.3 | 16.8 |
| STCN+CoVOS    | 79.0 | 79.4 | 83.6 | 72.6 | 80.4 | 57.9 |
```bash
# Youtube-VOS 2018, base model: stcn
CUDA_VISIBLE_DEVICES=0 scripts/exps/covos_yt2018_stcn.sh
```


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
