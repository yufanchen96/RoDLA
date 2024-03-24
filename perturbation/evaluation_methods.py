import os
import numpy as np
import torch
from pyiqa.archs.mad_arch import MAD
from pyiqa.archs.ssim_arch import ms_ssim, CW_SSIM
import cv2
import argparse
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, convert_to_coco_json
import copy
import cv2
import numpy as np
from content import add_watermark, add_image_background
from noise import add_blotch_noise, add_fibrous_noise
from blur import apply_gaussian_blur, apply_motion_blur
from inconsistency import apply_erosion, apply_dilation, apply_uneven_brightness
from spatial import apply_rotation, apply_perspective_transform, apply_elastic_transform
import json

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def traverse_and_create_directory(path):
    path_parts = path.split(os.path.sep)[:-1]
    current_directory = ""
    for part in path_parts:
        current_directory = os.path.join(current_directory, part)
        if current_directory == "":
            continue
        create_directory_if_not_exists(current_directory)

def compute_psnr(img, img_pert):
    if img.shape != img_pert.shape:
        img_pert = cv2.resize(img_pert, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(img.shape) == 3:
        mse = np.mean(np.square(img - img_pert), axis=(0, 1))
    else:
        mse = np.mean(np.square(img - img_pert))
    if np.any(mse == 0):
        psnr = np.inf
    else:
        psnr = np.mean(10 * np.log10(255 * 255 / mse))
    return psnr

def compute_ms_ssim(img, img_pert):
    if img.shape != img_pert.shape:
        img_pert = cv2.resize(img_pert, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    img_pert = torch.tensor(img_pert, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    return ms_ssim(img, img_pert, data_range=255).item()

def compute_cw_ssim(img, img_pert):
    if img.shape != img_pert.shape:
        img_pert = cv2.resize(img_pert, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    img_pert = torch.tensor(img_pert, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    return CW_SSIM().cw_ssim(img, img_pert, test_y_channel=True).item()

class perturbation_evaluator:
    def __init__(self, original_data_dir, original_json_dir):
        register_coco_instances(
            "original",
            {},
            original_json_dir,
            original_data_dir
        )
        self.metadata = copy.deepcopy(MetadataCatalog.get("original"))
        self.dataset_dicts = copy.deepcopy(DatasetCatalog.get("original"))

    def evaluate(self, pert_data_dir, pert_json_dir, metric):
        register_coco_instances(
            "pert",
            {},
            pert_json_dir,
            pert_data_dir
        )
        pert_dataset_dicts = copy.deepcopy(DatasetCatalog.get("pert"))
        assert len(self.dataset_dicts) == len(pert_dataset_dicts)
        result_list = []
        evaluation_dataset = []
        for i in range(len(self.dataset_dicts)):
            img = cv2.imread(self.dataset_dicts[i]["file_name"])
            img_pert = cv2.imread(pert_dataset_dicts[i]["file_name"])
            if metric == "psnr":
                result = compute_psnr(img, img_pert)
            elif metric == "ms_ssim":
                result = compute_ms_ssim(img, img_pert)
            elif metric == "cw_ssim":
                result = compute_cw_ssim(img, img_pert)
            else:
                raise NotImplementedError
            pert_dataset_dicts[i]["metric"] = result
            distorted_data = {
                "file_name":pert_dataset_dicts[i]["file_name"].split("/")[-1],
                "image_id": pert_dataset_dicts[i]["image_id"],
                metric: result
            }
            evaluation_dataset.append(distorted_data)
            result_list.append(result)
        metric_save_path = os.path.join("/".join(pert_json_dir.split("/")[:-1]), metric + ".json")
        with open(metric_save_path, "w") as f:
            json.dump(evaluation_dataset, f)
        result_average = np.mean(np.array(result_list))
        print("Average {} is {}".format(metric, result_average))
        with open(os.path.join("/".join(pert_json_dir.split("/")[:-1]), "average_" + metric + ".txt"), "w") as f:
            f.write("Average {} is {}".format(metric, result_average))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculation of image quality metrics for perturbed images.')
    parser.add_argument("--ori_dir")
    parser.add_argument("--ori_json_dir")
    parser.add_argument("--pert_dir")
    parser.add_argument("--pert_json_dir")
    parser.add_argument("--metric")
    args = parser.parse_args()
    print("Command Line Args:", args)
    pert_eval = perturbation_evaluator(args.ori_dir, args.ori_json_dir)
    pert_eval.evaluate(args.pert_dir, args.pert_json_dir, args.metric)
