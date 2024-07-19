import argparse
import os
import cv2
import torch
import numpy as np
from generate_weight import generate_weight, generate_connection
from evaluator import SaliencyEvaluator
from logger import Logger
import yaml
from tqdm import tqdm

# python main.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start!")
    parser.add_argument('--config',     type=str,          default='config.yaml', help="path to the config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if not os.path.exists(os.path.join(config["log_dir"], config["exp_name"])):
        os.makedirs(os.path.join(config["log_dir"], config["exp_name"]))
    log = Logger(os.path.join(config["log_dir"], config["exp_name"]))

    # generate weight mask
    gt_path = os.path.join(config["root_path"], config["mask"])
    pred_path = config["pred_path"]
    weight_path = os.path.join(config["root_path"], config["weight"])
    connection_path = os.path.join(config["root_path"], config["connection"])
    assert os.path.exists(gt_path), "ground truth mask path does not exist!"
    assert os.path.exists(pred_path), "prediciton mask path does not exist!"
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    if not os.path.exists(connection_path):
        os.mkdir(connection_path)

    if config["generate_weight"]:
        generate_connection(config)
        generate_weight(config)

    Evaluator = SaliencyEvaluator()
    metrics = config["metrics"]

    pred_list = os.listdir(pred_path)
    for pred_item in tqdm(pred_list, desc="Evaluation ..."):
        name = pred_item.rsplit('.')[0]
        pred = cv2.imread(os.path.join(pred_path, pred_item), 0)
        gt = cv2.imread(os.path.join(gt_path, name + config["mask_postfix"]), 0) # prediction mask is ended with png
        weight = np.load(os.path.join(weight_path, name + '.npy')) # weight mask is ended with npy

        pred = cv2.resize(pred, (config["image_size"], config["image_size"]))
        gt = cv2.resize(gt, (config["image_size"], config["image_size"]))
        weight = cv2.resize(weight, (config["image_size"], config["image_size"]))

        if config["normalize"]:
            pred = torch.from_numpy(pred / 255.0).unsqueeze(0) # may be lose some precision, because np.round(pred * 255) was used when saving the predicted maps
        gt = torch.from_numpy(gt / 255.0).unsqueeze(0)
        weight = torch.from_numpy(weight).unsqueeze(0)

        Evaluator.add_batch(pred, gt, metrics, weight, True) # scale=True. Scale=False equals to alpha=0.

    mae, si_mae, avg_auc, si_auc, mean_F, max_F, si_mean_F, si_max_F, Em = Evaluator.get_result(metrics)

    # print(f"mae:{mae}, si_mae:{si_mae}, auc:{avg_auc}, si_auc:{si_auc}, mean_F:{mean_F}, si_mean_f:{si_mean_F}, max_F:{max_F}, si_max_f:{si_max_F}, Em:{Em}")
    log.info("=======================================================")
    log.info("         Evaluation Result")
    log.info("=======================================================")
    log.info(f"        mae:{mae:.4f}")
    log.info(f"        si-mae:{si_mae:.4f}")
    log.info(f"        auc:{avg_auc:.4f}")
    log.info(f"        si-auc:{si_auc:.4f}")
    log.info(f"        mean_F:{mean_F:.4f}")
    log.info(f"        si-mean_F:{si_mean_F:.4f}")
    log.info(f"        max_F:{max_F:.4f}")
    log.info(f"        si-max_F:{si_max_F:.4f}")
    log.info(f"        Em:{Em:.4f}")
    log.info("=======================================================")




