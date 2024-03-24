from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, convert_to_coco_json
import copy
import cv2
import numpy as np
from content import add_watermark, add_image_background
from noise import apply_speckle, apply_texture
from blur import apply_defocus, apply_vibration
from inconsistency import apply_ink_holdout, apply_ink_bleeding, apply_illumination
from spatial import apply_rotation, apply_keystoning, apply_warping
import os
import json
from evaluation_methods import traverse_and_create_directory, compute_psnr, compute_ms_ssim, compute_cw_ssim
import argparse


def apply_augmentation(image, degree, enhance, dataset_dict, cfg):
    if enhance == "none":
        return image
    if enhance == "watermark" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) +'/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = add_watermark(image, degree=degree, save_path=save_path)
    if enhance == "speckle" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = apply_speckle(image, degree=degree, save_path=save_path)
    if enhance == "defocus" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = apply_defocus(image, degree=degree, save_path=save_path)
    if enhance == "vibration" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = apply_vibration(image, degree=degree, save_path=save_path)
    if enhance == "texture" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = apply_texture(image, degree=degree, save_path=save_path)
    if enhance == "background" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = add_image_background(image, degree=degree, save_path=save_path)
    if enhance == "ink_holdout" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = apply_ink_holdout(image, degree=degree, save_path=save_path)
    if enhance == "ink_bleeding" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = apply_ink_bleeding(image, degree=degree, save_path=save_path)
    if enhance == "illumination" and degree != 0:
        save_path = '/'.join(dataset_dict["file_name"].split("/")[:-3])
        save_path = os.path.join(save_path, cfg.AUG.ENHANCE + "_" + str(degree)) + '/' + '/'.join(
            dataset_dict["file_name"].split("/")[-2:])
        traverse_and_create_directory(save_path)
        image = apply_illumination(image, degree=degree, save_path=save_path)
    return image


class perturbation_data_creator:
    def __init__(self, dataset_dir, json_dir, dataset_name, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        register_coco_instances(
            dataset_name,
            {},
            json_dir,
            dataset_dir
        )
        self.metadata = copy.deepcopy(MetadataCatalog.get(dataset_name))
        self.dataset_dicts = copy.deepcopy(DatasetCatalog.get(dataset_name))
        self.dataset_name = dataset_name


    def create_pert_dataset(self, pert_method, degree, metric=None, background_folder=None):
        distorted_dataset = []
        distorted_dataset_name = self.dataset_name + "_" + pert_method + "_" + str(degree)
        distorted_root_path = (
            os.path.join(self.output_dir, os.path.join(pert_method, os.path.join(pert_method + "_" + str(degree),
                                                       self.metadata.get("image_root").split("/")[-1]))))
        traverse_and_create_directory(distorted_root_path)
        os.makedirs(distorted_root_path, exist_ok=True)
        json_path = (os.path.join(self.output_dir, 
                                  os.path.join(pert_method, 
                                               os.path.join(pert_method + "_" + str(degree), 
                                                            self.metadata.get("json_file").split("/")[-1]))))
        meta_path = json_path.split(".")[0] + "_meta" + ".json"
        iqa_metric_dict = {"psnr": [], "mad": [], "ms_ssim": [], "cw_ssim": []}
        for idx, data in enumerate(self.dataset_dicts):
            image = cv2.imread(data["file_name"])
            annos = copy.deepcopy(data["annotations"])
            pert_img, pert_annos = self.perturbation(image, annos, pert_method, degree, background_folder)
            psnr_metric, mad_metric, ms_ssim_metric, cw_ssim_metric = None, None, None, None
            if metric is not None:
                if metric == "psnr" or metric == "all":
                    psnr_metric = compute_psnr(image, pert_img)
                    iqa_metric_dict["psnr"].append(psnr_metric)
                if metric == "ms_ssim" or metric == "all":
                    ms_ssim_metric = compute_ms_ssim(image, pert_img)
                    iqa_metric_dict["ms_ssim"].append(ms_ssim_metric)
                if metric == "cw_ssim" or metric == "all":
                    cw_ssim_metric = compute_cw_ssim(image, pert_img)
                    iqa_metric_dict["cw_ssim"].append(cw_ssim_metric)
            image_name = data["file_name"].split("/")[-1]
            file_name = os.path.join(distorted_root_path, image_name)
            cv2.imwrite(file_name, pert_img)
            perturbed_image = cv2.imread(file_name)
            perturbed_height, perturbed_width = perturbed_image.shape[:2]
            distorted_data = {
                "file_name": data["file_name"].split("/")[-1],
                "image_id": data["image_id"],
                "height": perturbed_height,
                "width": perturbed_width,
                "annotations": pert_annos,
                "iqa_metric": {"psnr": psnr_metric,
                               "ms_ssim": ms_ssim_metric,
                               "cw_ssim": cw_ssim_metric}
            }
            distorted_dataset.append(distorted_data)
            if idx % 100 == 0:
                print("{} {} Perturbation Process: ".format(pert_method, str(degree)), idx + 1, "/", len(self.dataset_dicts))
        iqa_metric_average = {"psnr": [], "ms_ssim": [], "cw_ssim": []}
        for key in iqa_metric_dict.keys():
            iqa_metric_dict[key] = np.array(iqa_metric_dict[key])
            if len(iqa_metric_dict[key]) != 0:
                iqa_metric_average[key] = np.mean(iqa_metric_dict[key])
            else:
                iqa_metric_average[key] = None
        print("Perturbation_iqa_metric_average: ", iqa_metric_average)
        DatasetCatalog.register(distorted_dataset_name, lambda: distorted_dataset)
        MetadataCatalog.get(distorted_dataset_name).set(
            image_root=distorted_root_path,
            json_file_path=json_path,
            evaluator_type="coco",
            thing_classes=MetadataCatalog.get(self.dataset_name).get("thing_classes"),
            thing_dataset_id_to_contiguous_id=MetadataCatalog.get(self.dataset_name).get("thing_dataset_id_to_contiguous_id"),
            iqa_metric=iqa_metric_average
        )
        metadata = MetadataCatalog.get(distorted_dataset_name).as_dict()
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        if os.path.exists(json_path):
            os.remove(json_path) 
        convert_to_coco_json(distorted_dataset_name, json_path)

    @staticmethod
    def perturbation(image, annos, pert, degree, background_folder=None):
        if pert == "none" and degree == 0:
            return image, annos
        elif pert == "warping":
            pert_image, pert_annos = apply_warping(image, annos=annos, degree=degree)
        elif pert == "keystoning":
            pert_image, pert_annos = apply_keystoning(image, annos=annos, degree=degree)
        elif pert == "rotation":
            pert_image, pert_annos = apply_rotation(image, annos=annos, degree=degree)
        else:
            pert_annos = annos
            if pert == "watermark":
                pert_image = add_watermark(image, degree=degree)
            elif pert == "speckle":
                pert_image = apply_speckle(image, degree=degree)
            elif pert == "defocus":
                pert_image = apply_defocus(image, degree=degree)
            elif pert == "vibration":
                pert_image = apply_vibration(image, degree=degree)
            elif pert == "texture":
                pert_image = apply_texture(image, degree=degree)
            elif pert == "background":
                pert_image = add_image_background(image, degree=degree, background_folder=background_folder)
            elif pert == "ink_holdout":
                pert_image = apply_ink_holdout(image, degree=degree)
            elif pert == "ink_bleeding":
                pert_image = apply_ink_bleeding(image, degree=degree)
            elif pert == "illumination":
                pert_image = apply_illumination(image, degree=degree)
            else:
                raise ValueError("Perturbation not in the list, should be one of the following: "
                                 "watermark, speckle, defocus, vibration, texture, "
                                 "background, ink_holdout, ink_bleeding, illumination, "
                                 "warping, keystoning, rotation")
        return pert_image, pert_annos



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create perturbation dataset.')
    parser.add_argument("--dataset_dir")
    parser.add_argument("--json_dir")
    parser.add_argument("--dataset_name")
    parser.add_argument("--output_dir")
    parser.add_argument("--pert_method")
    parser.add_argument("--degree")
    parser.add_argument("--background_folder", default=None)
    parser.add_argument("--metric", default=None)
    args = parser.parse_args()
    print("Command Line Args:", args)
    if args.pert_method == "none":
        raise ValueError("Perturbation method should not be none.")
    elif args.pert_method == "all":
        key_list = ["rotation", "warping", "keystoning", "watermark", "speckle", "defocus",
                    "vibration", "texture", "background", "ink_holdout", "ink_bleeding", "illumination"]
        degree_list = [1, 2, 3]
    else:
        degree = int(args.degree)
        key_list = [args.pert_method, ]
        degree_list = [degree, ]
    if args.metric is not None:
        if args.metric not in ["psnr", "ms_ssim", "cw_ssim", "all"]:
            raise ValueError("Metric should be one of the following: psnr, ms_ssim, cw_ssim, all")
    data_perturbation = perturbation_data_creator(args.dataset_dir, args.json_dir, args.dataset_name, args.output_dir)
    for key in key_list:
        for degree in degree_list:
            if key == "background":
                data_perturbation.create_pert_dataset(key, degree, metric=args.metric, background_folder=args.background_folder)
            else:
                data_perturbation.create_pert_dataset(key, degree, metric=args.metric)
