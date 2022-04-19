# https://github.com/andr345/frtm-vos

import os
import json

from pathlib import Path

from lib.datasets import DAVISDataset
from lib.evaluation import evaluate_dataset
from path_config import path_config

if __name__ == "__main__":
    davis_path=os.path.join(path_config.data_path(), 'DAVIS')
    datasets = dict(
        dv2016val=(DAVISDataset, dict(path=davis_path, year="2016", split="val")),
        dv2017val=(DAVISDataset, dict(path=davis_path, year="2017", split="val")),
    )

    save_path = os.environ.get("RESULT_PATH")
    dset = os.environ.get("DSET")
    keyframe_json = os.environ.get("KEYFRAME_JSON")
    print(save_path, keyframe_json)
    if keyframe_json:
        keyframe_json = os.path.join(save_path, keyframe_json)
        with open(keyframe_json) as json_file:
            keyframe_dict = json.load(json_file)
    else:
        keyframe_dict = None

    dset = datasets[dset]  # DAVIS or YT dataset
    dset = dset[0](**dset[1])
    out_path = Path(save_path)

    dset.all_annotations = True

    print("Computing J-scores")
    evaluate_dataset(
        dset, keyframe_dict, out_path, measure="J"
    )  # print evaluation results
    print()
    print("Computing F-scores")
    evaluate_dataset(
        dset, keyframe_dict, out_path, measure="F"
    )  # print evaluation results
